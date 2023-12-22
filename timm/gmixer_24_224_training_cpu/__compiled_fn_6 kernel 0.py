
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


cpp_fused_div_native_layer_norm_backward_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1000L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp7 = tmp5 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr2[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                        auto tmp14 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = static_cast<float>(196.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp7 = static_cast<float>(384.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 - tmp11;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp12 - tmp16;
                        auto tmp18 = at::vec::Vectorized<float>(tmp0);
                        auto tmp19 = tmp18 * tmp17;
                        tmp19.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp1 = static_cast<float>(196.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp3;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (1536L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(1536);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-768L) + x1 + (768L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-768L) + x1 + (1536L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (1536L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((384L*x1) + (384L*x1_inner) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>((384L*x1) + (75264L*(c10::div_floor_integer(x0, 384L))) + (static_cast<long>(x0) % static_cast<long>(384L)))];
                    out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_sum_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (196L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(192L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (196L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (384L*x0))];
                        auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp12 = tmp0 >= tmp3;
                    auto tmp13 = static_cast<long>(384);
                    auto tmp14 = tmp0 < tmp13;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr1[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>((-192L) + x1 + (384L*x0))];
                        auto tmp20 = decltype(tmp19)(1) / (decltype(tmp19)(1) + std::exp(-tmp19));
                        auto tmp21 = static_cast<float>(1.0);
                        auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                        auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                        auto tmp24 = decltype(tmp23)(tmp23 + tmp21);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp28 = tmp4 ? tmp11 : tmp27;
                    out_ptr1[static_cast<long>(x1 + (384L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_layer_norm_backward_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)), static_cast<long>(384L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (75264L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x2 + x2_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp6 = tmp3 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp6;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (75264L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (75264L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (75264L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (75264L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (75264L*x0)), static_cast<long>(196L), tmp2, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp9 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                            auto tmp13 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 - tmp10;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp11 - tmp15;
                            auto tmp17 = at::vec::Vectorized<float>(tmp1);
                            auto tmp18 = tmp17 * tmp16;
                            auto tmp19 = tmp0 + tmp18;
                            tmp19.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (75264L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2)];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (75264L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                        auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                        auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                        in_out_ptr0[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp14;
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_159, primals_165, primals_171, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_231, primals_237, primals_243, primals_249, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, primals_295, mul, view_1, getitem_2, getitem_3, view_3, mul_4, view_5, getitem_6, getitem_7, view_7, mul_8, view_9, getitem_10, getitem_11, view_11, mul_12, view_13, getitem_14, getitem_15, view_15, mul_16, view_17, getitem_18, getitem_19, view_19, mul_20, view_21, getitem_22, getitem_23, view_23, mul_24, view_25, getitem_26, getitem_27, view_27, mul_28, view_29, getitem_30, getitem_31, view_31, mul_32, view_33, getitem_34, getitem_35, view_35, mul_36, view_37, getitem_38, getitem_39, view_39, mul_40, view_41, getitem_42, getitem_43, view_43, mul_44, view_45, getitem_46, getitem_47, view_47, mul_48, view_49, getitem_50, getitem_51, view_51, mul_52, view_53, getitem_54, getitem_55, view_55, mul_56, view_57, getitem_58, getitem_59, view_59, mul_60, view_61, getitem_62, getitem_63, view_63, mul_64, view_65, getitem_66, getitem_67, view_67, mul_68, view_69, getitem_70, getitem_71, view_71, mul_72, view_73, getitem_74, getitem_75, view_75, mul_76, view_77, getitem_78, getitem_79, view_79, mul_80, view_81, getitem_82, getitem_83, view_83, mul_84, view_85, getitem_86, getitem_87, view_87, mul_88, view_89, getitem_90, getitem_91, view_91, mul_92, view_93, getitem_94, getitem_95, view_95, mul_96, view_97, getitem_98, getitem_99, view_99, mul_100, view_101, getitem_102, getitem_103, view_103, mul_104, view_105, getitem_106, getitem_107, view_107, mul_108, view_109, getitem_110, getitem_111, view_111, mul_112, view_113, getitem_114, getitem_115, view_115, mul_116, view_117, getitem_118, getitem_119, view_119, mul_120, view_121, getitem_122, getitem_123, view_123, mul_124, view_125, getitem_126, getitem_127, view_127, mul_128, view_129, getitem_130, getitem_131, view_131, mul_132, view_133, getitem_134, getitem_135, view_135, mul_136, view_137, getitem_138, getitem_139, view_139, mul_140, view_141, getitem_142, getitem_143, view_143, mul_144, view_145, getitem_146, getitem_147, view_147, mul_148, view_149, getitem_150, getitem_151, view_151, mul_152, view_153, getitem_154, getitem_155, view_155, mul_156, view_157, getitem_158, getitem_159, view_159, mul_160, view_161, getitem_162, getitem_163, view_163, mul_164, view_165, getitem_166, getitem_167, view_167, mul_168, view_169, getitem_170, getitem_171, view_171, mul_172, view_173, getitem_174, getitem_175, view_175, mul_176, view_177, getitem_178, getitem_179, view_179, mul_180, view_181, getitem_182, getitem_183, view_183, mul_184, view_185, getitem_186, getitem_187, view_187, mul_188, view_189, getitem_190, getitem_191, view_191, mul_192, clone_169, permute_146, div_1, permute_150, permute_155, div_2, permute_160, permute_167, div_3, permute_170, permute_175, div_4, permute_180, permute_187, div_5, permute_190, permute_195, div_6, permute_200, permute_207, div_7, permute_210, permute_215, div_8, permute_220, permute_227, div_9, permute_230, permute_235, div_10, permute_240, permute_247, div_11, permute_250, permute_255, div_12, permute_260, permute_267, div_13, permute_270, permute_275, div_14, permute_280, permute_287, div_15, permute_290, permute_295, div_16, permute_300, permute_307, div_17, permute_310, permute_315, div_18, permute_320, permute_327, div_19, permute_330, permute_335, div_20, permute_340, permute_347, div_21, permute_350, permute_355, div_22, permute_360, permute_367, div_23, permute_370, permute_375, div_24, permute_380, permute_387, div_25, permute_390, permute_395, div_26, permute_400, permute_407, div_27, permute_410, permute_415, div_28, permute_420, permute_427, div_29, permute_430, permute_435, div_30, permute_440, permute_447, div_31, permute_450, permute_455, div_32, permute_460, permute_467, div_33, permute_470, permute_475, div_34, permute_480, permute_487, div_35, permute_490, permute_495, div_36, permute_500, permute_507, div_37, permute_510, permute_515, div_38, permute_520, permute_527, div_39, permute_530, permute_535, div_40, permute_540, permute_547, div_41, permute_550, permute_555, div_42, permute_560, permute_567, div_43, permute_570, permute_575, div_44, permute_580, permute_587, div_45, permute_590, permute_595, div_46, permute_600, permute_607, div_47, permute_610, permute_615, div_48, permute_620, permute_627, div_49, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (384, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_3, (384, ), (1, ))
    assert_size_stride(primals_9, (384, ), (1, ))
    assert_size_stride(primals_15, (384, ), (1, ))
    assert_size_stride(primals_21, (384, ), (1, ))
    assert_size_stride(primals_27, (384, ), (1, ))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_255, (384, ), (1, ))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_267, (384, ), (1, ))
    assert_size_stride(primals_273, (384, ), (1, ))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_295, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_1, (3072, 196), (196, 1))
    assert_size_stride(getitem_2, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_3, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_3, (3072, 192), (192, 1))
    assert_size_stride(mul_4, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_5, (1568, 384), (384, 1))
    assert_size_stride(getitem_6, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_7, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_7, (1568, 768), (768, 1))
    assert_size_stride(mul_8, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_9, (3072, 196), (196, 1))
    assert_size_stride(getitem_10, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_11, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_11, (3072, 192), (192, 1))
    assert_size_stride(mul_12, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_13, (1568, 384), (384, 1))
    assert_size_stride(getitem_14, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_15, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_15, (1568, 768), (768, 1))
    assert_size_stride(mul_16, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_17, (3072, 196), (196, 1))
    assert_size_stride(getitem_18, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_19, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_19, (3072, 192), (192, 1))
    assert_size_stride(mul_20, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_21, (1568, 384), (384, 1))
    assert_size_stride(getitem_22, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_23, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_23, (1568, 768), (768, 1))
    assert_size_stride(mul_24, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_25, (3072, 196), (196, 1))
    assert_size_stride(getitem_26, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_27, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_27, (3072, 192), (192, 1))
    assert_size_stride(mul_28, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_29, (1568, 384), (384, 1))
    assert_size_stride(getitem_30, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_31, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_31, (1568, 768), (768, 1))
    assert_size_stride(mul_32, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_33, (3072, 196), (196, 1))
    assert_size_stride(getitem_34, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_35, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_35, (3072, 192), (192, 1))
    assert_size_stride(mul_36, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_37, (1568, 384), (384, 1))
    assert_size_stride(getitem_38, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_39, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_39, (1568, 768), (768, 1))
    assert_size_stride(mul_40, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_41, (3072, 196), (196, 1))
    assert_size_stride(getitem_42, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_43, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_43, (3072, 192), (192, 1))
    assert_size_stride(mul_44, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_45, (1568, 384), (384, 1))
    assert_size_stride(getitem_46, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_47, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_47, (1568, 768), (768, 1))
    assert_size_stride(mul_48, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_49, (3072, 196), (196, 1))
    assert_size_stride(getitem_50, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_51, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_51, (3072, 192), (192, 1))
    assert_size_stride(mul_52, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_53, (1568, 384), (384, 1))
    assert_size_stride(getitem_54, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_55, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_55, (1568, 768), (768, 1))
    assert_size_stride(mul_56, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_57, (3072, 196), (196, 1))
    assert_size_stride(getitem_58, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_59, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_59, (3072, 192), (192, 1))
    assert_size_stride(mul_60, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_61, (1568, 384), (384, 1))
    assert_size_stride(getitem_62, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_63, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_63, (1568, 768), (768, 1))
    assert_size_stride(mul_64, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_65, (3072, 196), (196, 1))
    assert_size_stride(getitem_66, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_67, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_67, (3072, 192), (192, 1))
    assert_size_stride(mul_68, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_69, (1568, 384), (384, 1))
    assert_size_stride(getitem_70, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_71, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_71, (1568, 768), (768, 1))
    assert_size_stride(mul_72, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_73, (3072, 196), (196, 1))
    assert_size_stride(getitem_74, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_75, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_75, (3072, 192), (192, 1))
    assert_size_stride(mul_76, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_77, (1568, 384), (384, 1))
    assert_size_stride(getitem_78, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_79, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_79, (1568, 768), (768, 1))
    assert_size_stride(mul_80, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_81, (3072, 196), (196, 1))
    assert_size_stride(getitem_82, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_83, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_83, (3072, 192), (192, 1))
    assert_size_stride(mul_84, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_85, (1568, 384), (384, 1))
    assert_size_stride(getitem_86, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_87, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_87, (1568, 768), (768, 1))
    assert_size_stride(mul_88, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_89, (3072, 196), (196, 1))
    assert_size_stride(getitem_90, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_91, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_91, (3072, 192), (192, 1))
    assert_size_stride(mul_92, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_93, (1568, 384), (384, 1))
    assert_size_stride(getitem_94, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_95, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_95, (1568, 768), (768, 1))
    assert_size_stride(mul_96, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_97, (3072, 196), (196, 1))
    assert_size_stride(getitem_98, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_99, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_99, (3072, 192), (192, 1))
    assert_size_stride(mul_100, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_101, (1568, 384), (384, 1))
    assert_size_stride(getitem_102, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_103, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_103, (1568, 768), (768, 1))
    assert_size_stride(mul_104, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_105, (3072, 196), (196, 1))
    assert_size_stride(getitem_106, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_107, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_107, (3072, 192), (192, 1))
    assert_size_stride(mul_108, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_109, (1568, 384), (384, 1))
    assert_size_stride(getitem_110, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_111, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_111, (1568, 768), (768, 1))
    assert_size_stride(mul_112, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_113, (3072, 196), (196, 1))
    assert_size_stride(getitem_114, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_115, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_115, (3072, 192), (192, 1))
    assert_size_stride(mul_116, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_117, (1568, 384), (384, 1))
    assert_size_stride(getitem_118, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_119, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_119, (1568, 768), (768, 1))
    assert_size_stride(mul_120, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_121, (3072, 196), (196, 1))
    assert_size_stride(getitem_122, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_123, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_123, (3072, 192), (192, 1))
    assert_size_stride(mul_124, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_125, (1568, 384), (384, 1))
    assert_size_stride(getitem_126, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_127, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_127, (1568, 768), (768, 1))
    assert_size_stride(mul_128, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_129, (3072, 196), (196, 1))
    assert_size_stride(getitem_130, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_131, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_131, (3072, 192), (192, 1))
    assert_size_stride(mul_132, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_133, (1568, 384), (384, 1))
    assert_size_stride(getitem_134, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_135, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_135, (1568, 768), (768, 1))
    assert_size_stride(mul_136, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_137, (3072, 196), (196, 1))
    assert_size_stride(getitem_138, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_139, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_139, (3072, 192), (192, 1))
    assert_size_stride(mul_140, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_141, (1568, 384), (384, 1))
    assert_size_stride(getitem_142, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_143, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_143, (1568, 768), (768, 1))
    assert_size_stride(mul_144, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_145, (3072, 196), (196, 1))
    assert_size_stride(getitem_146, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_147, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_147, (3072, 192), (192, 1))
    assert_size_stride(mul_148, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_149, (1568, 384), (384, 1))
    assert_size_stride(getitem_150, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_151, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_151, (1568, 768), (768, 1))
    assert_size_stride(mul_152, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_153, (3072, 196), (196, 1))
    assert_size_stride(getitem_154, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_155, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_155, (3072, 192), (192, 1))
    assert_size_stride(mul_156, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_157, (1568, 384), (384, 1))
    assert_size_stride(getitem_158, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_159, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_159, (1568, 768), (768, 1))
    assert_size_stride(mul_160, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_161, (3072, 196), (196, 1))
    assert_size_stride(getitem_162, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_163, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_163, (3072, 192), (192, 1))
    assert_size_stride(mul_164, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_165, (1568, 384), (384, 1))
    assert_size_stride(getitem_166, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_167, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_167, (1568, 768), (768, 1))
    assert_size_stride(mul_168, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_169, (3072, 196), (196, 1))
    assert_size_stride(getitem_170, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_171, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_171, (3072, 192), (192, 1))
    assert_size_stride(mul_172, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_173, (1568, 384), (384, 1))
    assert_size_stride(getitem_174, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_175, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_175, (1568, 768), (768, 1))
    assert_size_stride(mul_176, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_177, (3072, 196), (196, 1))
    assert_size_stride(getitem_178, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_179, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_179, (3072, 192), (192, 1))
    assert_size_stride(mul_180, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_181, (1568, 384), (384, 1))
    assert_size_stride(getitem_182, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_183, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_183, (1568, 768), (768, 1))
    assert_size_stride(mul_184, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_185, (3072, 196), (196, 1))
    assert_size_stride(getitem_186, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_187, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_187, (3072, 192), (192, 1))
    assert_size_stride(mul_188, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_189, (1568, 384), (384, 1))
    assert_size_stride(getitem_190, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_191, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_191, (1568, 768), (768, 1))
    assert_size_stride(mul_192, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(clone_169, (8, 384), (384, 1))
    assert_size_stride(permute_146, (1000, 384), (384, 1))
    assert_size_stride(div_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_150, (384, 768), (768, 1))
    assert_size_stride(permute_155, (1536, 384), (384, 1))
    assert_size_stride(div_2, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_160, (196, 192), (192, 1))
    assert_size_stride(permute_167, (384, 196), (196, 1))
    assert_size_stride(div_3, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_170, (384, 768), (768, 1))
    assert_size_stride(permute_175, (1536, 384), (384, 1))
    assert_size_stride(div_4, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_180, (196, 192), (192, 1))
    assert_size_stride(permute_187, (384, 196), (196, 1))
    assert_size_stride(div_5, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_190, (384, 768), (768, 1))
    assert_size_stride(permute_195, (1536, 384), (384, 1))
    assert_size_stride(div_6, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_200, (196, 192), (192, 1))
    assert_size_stride(permute_207, (384, 196), (196, 1))
    assert_size_stride(div_7, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_210, (384, 768), (768, 1))
    assert_size_stride(permute_215, (1536, 384), (384, 1))
    assert_size_stride(div_8, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_220, (196, 192), (192, 1))
    assert_size_stride(permute_227, (384, 196), (196, 1))
    assert_size_stride(div_9, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_230, (384, 768), (768, 1))
    assert_size_stride(permute_235, (1536, 384), (384, 1))
    assert_size_stride(div_10, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (196, 192), (192, 1))
    assert_size_stride(permute_247, (384, 196), (196, 1))
    assert_size_stride(div_11, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_250, (384, 768), (768, 1))
    assert_size_stride(permute_255, (1536, 384), (384, 1))
    assert_size_stride(div_12, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_260, (196, 192), (192, 1))
    assert_size_stride(permute_267, (384, 196), (196, 1))
    assert_size_stride(div_13, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_270, (384, 768), (768, 1))
    assert_size_stride(permute_275, (1536, 384), (384, 1))
    assert_size_stride(div_14, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_280, (196, 192), (192, 1))
    assert_size_stride(permute_287, (384, 196), (196, 1))
    assert_size_stride(div_15, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_290, (384, 768), (768, 1))
    assert_size_stride(permute_295, (1536, 384), (384, 1))
    assert_size_stride(div_16, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_300, (196, 192), (192, 1))
    assert_size_stride(permute_307, (384, 196), (196, 1))
    assert_size_stride(div_17, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_310, (384, 768), (768, 1))
    assert_size_stride(permute_315, (1536, 384), (384, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_320, (196, 192), (192, 1))
    assert_size_stride(permute_327, (384, 196), (196, 1))
    assert_size_stride(div_19, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_330, (384, 768), (768, 1))
    assert_size_stride(permute_335, (1536, 384), (384, 1))
    assert_size_stride(div_20, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_340, (196, 192), (192, 1))
    assert_size_stride(permute_347, (384, 196), (196, 1))
    assert_size_stride(div_21, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_350, (384, 768), (768, 1))
    assert_size_stride(permute_355, (1536, 384), (384, 1))
    assert_size_stride(div_22, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_360, (196, 192), (192, 1))
    assert_size_stride(permute_367, (384, 196), (196, 1))
    assert_size_stride(div_23, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_370, (384, 768), (768, 1))
    assert_size_stride(permute_375, (1536, 384), (384, 1))
    assert_size_stride(div_24, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_380, (196, 192), (192, 1))
    assert_size_stride(permute_387, (384, 196), (196, 1))
    assert_size_stride(div_25, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_390, (384, 768), (768, 1))
    assert_size_stride(permute_395, (1536, 384), (384, 1))
    assert_size_stride(div_26, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_400, (196, 192), (192, 1))
    assert_size_stride(permute_407, (384, 196), (196, 1))
    assert_size_stride(div_27, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_410, (384, 768), (768, 1))
    assert_size_stride(permute_415, (1536, 384), (384, 1))
    assert_size_stride(div_28, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_420, (196, 192), (192, 1))
    assert_size_stride(permute_427, (384, 196), (196, 1))
    assert_size_stride(div_29, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_430, (384, 768), (768, 1))
    assert_size_stride(permute_435, (1536, 384), (384, 1))
    assert_size_stride(div_30, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_440, (196, 192), (192, 1))
    assert_size_stride(permute_447, (384, 196), (196, 1))
    assert_size_stride(div_31, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_450, (384, 768), (768, 1))
    assert_size_stride(permute_455, (1536, 384), (384, 1))
    assert_size_stride(div_32, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_460, (196, 192), (192, 1))
    assert_size_stride(permute_467, (384, 196), (196, 1))
    assert_size_stride(div_33, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_470, (384, 768), (768, 1))
    assert_size_stride(permute_475, (1536, 384), (384, 1))
    assert_size_stride(div_34, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_480, (196, 192), (192, 1))
    assert_size_stride(permute_487, (384, 196), (196, 1))
    assert_size_stride(div_35, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_490, (384, 768), (768, 1))
    assert_size_stride(permute_495, (1536, 384), (384, 1))
    assert_size_stride(div_36, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_500, (196, 192), (192, 1))
    assert_size_stride(permute_507, (384, 196), (196, 1))
    assert_size_stride(div_37, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_510, (384, 768), (768, 1))
    assert_size_stride(permute_515, (1536, 384), (384, 1))
    assert_size_stride(div_38, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_520, (196, 192), (192, 1))
    assert_size_stride(permute_527, (384, 196), (196, 1))
    assert_size_stride(div_39, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_530, (384, 768), (768, 1))
    assert_size_stride(permute_535, (1536, 384), (384, 1))
    assert_size_stride(div_40, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_540, (196, 192), (192, 1))
    assert_size_stride(permute_547, (384, 196), (196, 1))
    assert_size_stride(div_41, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_550, (384, 768), (768, 1))
    assert_size_stride(permute_555, (1536, 384), (384, 1))
    assert_size_stride(div_42, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_560, (196, 192), (192, 1))
    assert_size_stride(permute_567, (384, 196), (196, 1))
    assert_size_stride(div_43, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_570, (384, 768), (768, 1))
    assert_size_stride(permute_575, (1536, 384), (384, 1))
    assert_size_stride(div_44, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_580, (196, 192), (192, 1))
    assert_size_stride(permute_587, (384, 196), (196, 1))
    assert_size_stride(div_45, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_590, (384, 768), (768, 1))
    assert_size_stride(permute_595, (1536, 384), (384, 1))
    assert_size_stride(div_46, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_600, (196, 192), (192, 1))
    assert_size_stride(permute_607, (384, 196), (196, 1))
    assert_size_stride(div_47, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_610, (384, 768), (768, 1))
    assert_size_stride(permute_615, (1536, 384), (384, 1))
    assert_size_stride(div_48, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_620, (196, 192), (192, 1))
    assert_size_stride(permute_627, (384, 196), (196, 1))
    assert_size_stride(div_49, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_146, out=buf0)
    del permute_146
    buf1 = empty((1000, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_169, out=buf1)
    del clone_169
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    buf6 = empty((384, ), device='cpu', dtype=torch.float32)
    buf7 = empty((384, ), device='cpu', dtype=torch.float32)
    cpp_fused_div_native_layer_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(mul_192.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf0
    del div_1
    del mul_192
    del primals_291
    del tangents_1
    buf8 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1568, 384), (384, 1), 0), permute_150, out=buf8)
    del permute_150
    buf9 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (384, 1568), (1, 384), 0), view_191, out=buf9)
    del view_191
    buf10 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf11 = empty((8, 196, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_1(c_void_p(buf5.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(getitem_191.data_ptr()), c_void_p(getitem_190.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del getitem_190
    del getitem_191
    buf12 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (1568, 1536), (1536, 1), 0), permute_155, out=buf12)
    del permute_155
    buf13 = empty((1536, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (1536, 1568), (1, 1536), 0), view_189, out=buf13)
    del view_189
    buf14 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf15 = buf4; del buf4  # reuse
    buf16 = buf3; del buf3  # reuse
    buf17 = empty((384, ), device='cpu', dtype=torch.float32)
    buf18 = empty((384, ), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf12, (8, 196, 384), (75264, 384, 1), 0); del buf12  # reuse
    buf20 = empty((3072, 196), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_2(c_void_p(buf19.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(mul_188.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()))
    del div_2
    del mul_188
    del primals_285
    buf21 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf20, permute_160, out=buf21)
    del permute_160
    buf22 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (196, 3072), (1, 196), 0), view_187, out=buf22)
    del view_187
    buf23 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf24 = empty((8, 384, 384), device='cpu', dtype=torch.float32)
    buf25 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_3(c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(getitem_187.data_ptr()), c_void_p(getitem_186.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del getitem_186
    del getitem_187
    buf26 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (384, 3072), (1, 384), 0), view_185, out=buf26)
    del view_185
    buf27 = buf20; del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (3072, 384), (384, 1), 0), permute_167, out=buf27)
    del permute_167
    buf28 = buf16; del buf16  # reuse
    buf29 = buf15; del buf15  # reuse
    buf30 = empty((384, ), device='cpu', dtype=torch.float32)
    buf31 = empty((384, ), device='cpu', dtype=torch.float32)
    buf32 = buf19; del buf19  # reuse
    cpp_fused_add_native_layer_norm_backward_4(c_void_p(buf32.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(mul_184.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del div_3
    del mul_184
    del primals_279
    buf33 = buf8; del buf8  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (1568, 384), (384, 1), 0), permute_170, out=buf33)
    del permute_170
    buf34 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (384, 1568), (1, 384), 0), view_183, out=buf34)
    del view_183
    buf35 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf36 = buf11; del buf11  # reuse
    cpp_fused_cat_sum_5(c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(getitem_183.data_ptr()), c_void_p(getitem_182.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del getitem_182
    del getitem_183
    buf37 = reinterpret_tensor(buf27, (1568, 384), (384, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (1568, 1536), (1536, 1), 0), permute_175, out=buf37)
    del permute_175
    buf38 = reinterpret_tensor(buf21, (1536, 384), (384, 1), 0); del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (1536, 1568), (1, 1536), 0), view_181, out=buf38)
    del view_181
    buf39 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf40 = buf29; del buf29  # reuse
    buf41 = buf28; del buf28  # reuse
    buf42 = empty((384, ), device='cpu', dtype=torch.float32)
    buf43 = empty((384, ), device='cpu', dtype=torch.float32)
    buf44 = buf32; del buf32  # reuse
    buf45 = reinterpret_tensor(buf5, (3072, 196), (196, 1), 0); del buf5  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_6(c_void_p(buf44.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(mul_180.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()))
    del div_4
    del mul_180
    del primals_273
    buf46 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf45, permute_180, out=buf46)
    del permute_180
    buf47 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (196, 3072), (1, 196), 0), view_179, out=buf47)
    del view_179
    buf48 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf49 = buf24; del buf24  # reuse
    buf50 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_7(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(getitem_179.data_ptr()), c_void_p(getitem_178.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    del getitem_178
    del getitem_179
    buf51 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (384, 3072), (1, 384), 0), view_177, out=buf51)
    del view_177
    buf52 = buf45; del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (3072, 384), (384, 1), 0), permute_187, out=buf52)
    del permute_187
    buf53 = buf41; del buf41  # reuse
    buf54 = buf40; del buf40  # reuse
    buf55 = empty((384, ), device='cpu', dtype=torch.float32)
    buf56 = empty((384, ), device='cpu', dtype=torch.float32)
    buf57 = buf44; del buf44  # reuse
    cpp_fused_add_native_layer_norm_backward_8(c_void_p(buf57.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(mul_176.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del div_5
    del mul_176
    del primals_267
    buf58 = buf33; del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (1568, 384), (384, 1), 0), permute_190, out=buf58)
    del permute_190
    buf59 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (384, 1568), (1, 384), 0), view_175, out=buf59)
    del view_175
    buf60 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf61 = buf36; del buf36  # reuse
    cpp_fused_cat_sum_9(c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(getitem_175.data_ptr()), c_void_p(getitem_174.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del getitem_174
    del getitem_175
    buf62 = reinterpret_tensor(buf52, (1568, 384), (384, 1), 0); del buf52  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (1568, 1536), (1536, 1), 0), permute_195, out=buf62)
    del permute_195
    buf63 = reinterpret_tensor(buf46, (1536, 384), (384, 1), 0); del buf46  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (1536, 1568), (1, 1536), 0), view_173, out=buf63)
    del view_173
    buf64 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf65 = buf54; del buf54  # reuse
    buf66 = buf53; del buf53  # reuse
    buf67 = empty((384, ), device='cpu', dtype=torch.float32)
    buf68 = empty((384, ), device='cpu', dtype=torch.float32)
    buf69 = buf57; del buf57  # reuse
    buf70 = reinterpret_tensor(buf37, (3072, 196), (196, 1), 0); del buf37  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_10(c_void_p(buf69.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(mul_172.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()))
    del div_6
    del mul_172
    del primals_261
    buf71 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf70, permute_200, out=buf71)
    del permute_200
    buf72 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (196, 3072), (1, 196), 0), view_171, out=buf72)
    del view_171
    buf73 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf74 = buf49; del buf49  # reuse
    buf75 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_11(c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(getitem_171.data_ptr()), c_void_p(getitem_170.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del getitem_170
    del getitem_171
    buf76 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (384, 3072), (1, 384), 0), view_169, out=buf76)
    del view_169
    buf77 = buf70; del buf70  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (3072, 384), (384, 1), 0), permute_207, out=buf77)
    del permute_207
    buf78 = buf66; del buf66  # reuse
    buf79 = buf65; del buf65  # reuse
    buf80 = empty((384, ), device='cpu', dtype=torch.float32)
    buf81 = empty((384, ), device='cpu', dtype=torch.float32)
    buf82 = buf69; del buf69  # reuse
    cpp_fused_add_native_layer_norm_backward_12(c_void_p(buf82.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(mul_168.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del div_7
    del mul_168
    del primals_255
    buf83 = buf58; del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (1568, 384), (384, 1), 0), permute_210, out=buf83)
    del permute_210
    buf84 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (384, 1568), (1, 384), 0), view_167, out=buf84)
    del view_167
    buf85 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf86 = buf61; del buf61  # reuse
    cpp_fused_cat_sum_13(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(getitem_167.data_ptr()), c_void_p(getitem_166.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del getitem_166
    del getitem_167
    buf87 = reinterpret_tensor(buf77, (1568, 384), (384, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (1568, 1536), (1536, 1), 0), permute_215, out=buf87)
    del permute_215
    buf88 = reinterpret_tensor(buf71, (1536, 384), (384, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (1536, 1568), (1, 1536), 0), view_165, out=buf88)
    del view_165
    buf89 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf90 = buf79; del buf79  # reuse
    buf91 = buf78; del buf78  # reuse
    buf92 = empty((384, ), device='cpu', dtype=torch.float32)
    buf93 = empty((384, ), device='cpu', dtype=torch.float32)
    buf94 = buf82; del buf82  # reuse
    buf95 = reinterpret_tensor(buf62, (3072, 196), (196, 1), 0); del buf62  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_14(c_void_p(buf94.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(mul_164.data_ptr()), c_void_p(div_8.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()))
    del div_8
    del mul_164
    del primals_249
    buf96 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf95, permute_220, out=buf96)
    del permute_220
    buf97 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (196, 3072), (1, 196), 0), view_163, out=buf97)
    del view_163
    buf98 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf99 = buf74; del buf74  # reuse
    buf100 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_15(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(getitem_163.data_ptr()), c_void_p(getitem_162.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    del getitem_162
    del getitem_163
    buf101 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (384, 3072), (1, 384), 0), view_161, out=buf101)
    del view_161
    buf102 = buf95; del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (3072, 384), (384, 1), 0), permute_227, out=buf102)
    del permute_227
    buf103 = buf91; del buf91  # reuse
    buf104 = buf90; del buf90  # reuse
    buf105 = empty((384, ), device='cpu', dtype=torch.float32)
    buf106 = empty((384, ), device='cpu', dtype=torch.float32)
    buf107 = buf94; del buf94  # reuse
    cpp_fused_add_native_layer_norm_backward_16(c_void_p(buf107.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(mul_160.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del div_9
    del mul_160
    del primals_243
    buf108 = buf83; del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (1568, 384), (384, 1), 0), permute_230, out=buf108)
    del permute_230
    buf109 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (384, 1568), (1, 384), 0), view_159, out=buf109)
    del view_159
    buf110 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf111 = buf86; del buf86  # reuse
    cpp_fused_cat_sum_17(c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(getitem_159.data_ptr()), c_void_p(getitem_158.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del getitem_158
    del getitem_159
    buf112 = reinterpret_tensor(buf102, (1568, 384), (384, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (1568, 1536), (1536, 1), 0), permute_235, out=buf112)
    del permute_235
    buf113 = reinterpret_tensor(buf96, (1536, 384), (384, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (1536, 1568), (1, 1536), 0), view_157, out=buf113)
    del view_157
    buf114 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf115 = buf104; del buf104  # reuse
    buf116 = buf103; del buf103  # reuse
    buf117 = empty((384, ), device='cpu', dtype=torch.float32)
    buf118 = empty((384, ), device='cpu', dtype=torch.float32)
    buf119 = buf107; del buf107  # reuse
    buf120 = reinterpret_tensor(buf87, (3072, 196), (196, 1), 0); del buf87  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_18(c_void_p(buf119.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(mul_156.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()))
    del div_10
    del mul_156
    del primals_237
    buf121 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf120, permute_240, out=buf121)
    del permute_240
    buf122 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf120, (196, 3072), (1, 196), 0), view_155, out=buf122)
    del view_155
    buf123 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf124 = buf99; del buf99  # reuse
    buf125 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_19(c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(getitem_155.data_ptr()), c_void_p(getitem_154.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    del getitem_154
    del getitem_155
    buf126 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (384, 3072), (1, 384), 0), view_153, out=buf126)
    del view_153
    buf127 = buf120; del buf120  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (3072, 384), (384, 1), 0), permute_247, out=buf127)
    del permute_247
    buf128 = buf116; del buf116  # reuse
    buf129 = buf115; del buf115  # reuse
    buf130 = empty((384, ), device='cpu', dtype=torch.float32)
    buf131 = empty((384, ), device='cpu', dtype=torch.float32)
    buf132 = buf119; del buf119  # reuse
    cpp_fused_add_native_layer_norm_backward_20(c_void_p(buf132.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(mul_152.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del div_11
    del mul_152
    del primals_231
    buf133 = buf108; del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (1568, 384), (384, 1), 0), permute_250, out=buf133)
    del permute_250
    buf134 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (384, 1568), (1, 384), 0), view_151, out=buf134)
    del view_151
    buf135 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf136 = buf111; del buf111  # reuse
    cpp_fused_cat_sum_21(c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(getitem_151.data_ptr()), c_void_p(getitem_150.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    del getitem_150
    del getitem_151
    buf137 = reinterpret_tensor(buf127, (1568, 384), (384, 1), 0); del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (1568, 1536), (1536, 1), 0), permute_255, out=buf137)
    del permute_255
    buf138 = reinterpret_tensor(buf121, (1536, 384), (384, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (1536, 1568), (1, 1536), 0), view_149, out=buf138)
    del view_149
    buf139 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf140 = buf129; del buf129  # reuse
    buf141 = buf128; del buf128  # reuse
    buf142 = empty((384, ), device='cpu', dtype=torch.float32)
    buf143 = empty((384, ), device='cpu', dtype=torch.float32)
    buf144 = buf132; del buf132  # reuse
    buf145 = reinterpret_tensor(buf112, (3072, 196), (196, 1), 0); del buf112  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_22(c_void_p(buf144.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(mul_148.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()))
    del div_12
    del mul_148
    del primals_225
    buf146 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf145, permute_260, out=buf146)
    del permute_260
    buf147 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (196, 3072), (1, 196), 0), view_147, out=buf147)
    del view_147
    buf148 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf149 = buf124; del buf124  # reuse
    buf150 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_23(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(getitem_147.data_ptr()), c_void_p(getitem_146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del getitem_146
    del getitem_147
    buf151 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (384, 3072), (1, 384), 0), view_145, out=buf151)
    del view_145
    buf152 = buf145; del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (3072, 384), (384, 1), 0), permute_267, out=buf152)
    del permute_267
    buf153 = buf141; del buf141  # reuse
    buf154 = buf140; del buf140  # reuse
    buf155 = empty((384, ), device='cpu', dtype=torch.float32)
    buf156 = empty((384, ), device='cpu', dtype=torch.float32)
    buf157 = buf144; del buf144  # reuse
    cpp_fused_add_native_layer_norm_backward_24(c_void_p(buf157.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(mul_144.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    del div_13
    del mul_144
    del primals_219
    buf158 = buf133; del buf133  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf157, (1568, 384), (384, 1), 0), permute_270, out=buf158)
    del permute_270
    buf159 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf157, (384, 1568), (1, 384), 0), view_143, out=buf159)
    del view_143
    buf160 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf161 = buf136; del buf136  # reuse
    cpp_fused_cat_sum_25(c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(getitem_143.data_ptr()), c_void_p(getitem_142.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del getitem_142
    del getitem_143
    buf162 = reinterpret_tensor(buf152, (1568, 384), (384, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (1568, 1536), (1536, 1), 0), permute_275, out=buf162)
    del permute_275
    buf163 = reinterpret_tensor(buf146, (1536, 384), (384, 1), 0); del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (1536, 1568), (1, 1536), 0), view_141, out=buf163)
    del view_141
    buf164 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf165 = buf154; del buf154  # reuse
    buf166 = buf153; del buf153  # reuse
    buf167 = empty((384, ), device='cpu', dtype=torch.float32)
    buf168 = empty((384, ), device='cpu', dtype=torch.float32)
    buf169 = buf157; del buf157  # reuse
    buf170 = reinterpret_tensor(buf137, (3072, 196), (196, 1), 0); del buf137  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_26(c_void_p(buf169.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(mul_140.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del div_14
    del mul_140
    del primals_213
    buf171 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf170, permute_280, out=buf171)
    del permute_280
    buf172 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf170, (196, 3072), (1, 196), 0), view_139, out=buf172)
    del view_139
    buf173 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf174 = buf149; del buf149  # reuse
    buf175 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_27(c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(getitem_139.data_ptr()), c_void_p(getitem_138.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del getitem_138
    del getitem_139
    buf176 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (384, 3072), (1, 384), 0), view_137, out=buf176)
    del view_137
    buf177 = buf170; del buf170  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (3072, 384), (384, 1), 0), permute_287, out=buf177)
    del permute_287
    buf178 = buf166; del buf166  # reuse
    buf179 = buf165; del buf165  # reuse
    buf180 = empty((384, ), device='cpu', dtype=torch.float32)
    buf181 = empty((384, ), device='cpu', dtype=torch.float32)
    buf182 = buf169; del buf169  # reuse
    cpp_fused_add_native_layer_norm_backward_28(c_void_p(buf182.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(mul_136.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del div_15
    del mul_136
    del primals_207
    buf183 = buf158; del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (1568, 384), (384, 1), 0), permute_290, out=buf183)
    del permute_290
    buf184 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (384, 1568), (1, 384), 0), view_135, out=buf184)
    del view_135
    buf185 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf186 = buf161; del buf161  # reuse
    cpp_fused_cat_sum_29(c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(getitem_135.data_ptr()), c_void_p(getitem_134.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del getitem_134
    del getitem_135
    buf187 = reinterpret_tensor(buf177, (1568, 384), (384, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (1568, 1536), (1536, 1), 0), permute_295, out=buf187)
    del permute_295
    buf188 = reinterpret_tensor(buf171, (1536, 384), (384, 1), 0); del buf171  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (1536, 1568), (1, 1536), 0), view_133, out=buf188)
    del view_133
    buf189 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf190 = buf179; del buf179  # reuse
    buf191 = buf178; del buf178  # reuse
    buf192 = empty((384, ), device='cpu', dtype=torch.float32)
    buf193 = empty((384, ), device='cpu', dtype=torch.float32)
    buf194 = buf182; del buf182  # reuse
    buf195 = reinterpret_tensor(buf162, (3072, 196), (196, 1), 0); del buf162  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_30(c_void_p(buf194.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(mul_132.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del div_16
    del mul_132
    del primals_201
    buf196 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf195, permute_300, out=buf196)
    del permute_300
    buf197 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (196, 3072), (1, 196), 0), view_131, out=buf197)
    del view_131
    buf198 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf199 = buf174; del buf174  # reuse
    buf200 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_31(c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(getitem_131.data_ptr()), c_void_p(getitem_130.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del getitem_130
    del getitem_131
    buf201 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (384, 3072), (1, 384), 0), view_129, out=buf201)
    del view_129
    buf202 = buf195; del buf195  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (3072, 384), (384, 1), 0), permute_307, out=buf202)
    del permute_307
    buf203 = buf191; del buf191  # reuse
    buf204 = buf190; del buf190  # reuse
    buf205 = empty((384, ), device='cpu', dtype=torch.float32)
    buf206 = empty((384, ), device='cpu', dtype=torch.float32)
    buf207 = buf194; del buf194  # reuse
    cpp_fused_add_native_layer_norm_backward_32(c_void_p(buf207.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(mul_128.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del div_17
    del mul_128
    del primals_195
    buf208 = buf183; del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (1568, 384), (384, 1), 0), permute_310, out=buf208)
    del permute_310
    buf209 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (384, 1568), (1, 384), 0), view_127, out=buf209)
    del view_127
    buf210 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf211 = buf186; del buf186  # reuse
    cpp_fused_cat_sum_33(c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(getitem_126.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del getitem_126
    del getitem_127
    buf212 = reinterpret_tensor(buf202, (1568, 384), (384, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (1568, 1536), (1536, 1), 0), permute_315, out=buf212)
    del permute_315
    buf213 = reinterpret_tensor(buf196, (1536, 384), (384, 1), 0); del buf196  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (1536, 1568), (1, 1536), 0), view_125, out=buf213)
    del view_125
    buf214 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf215 = buf204; del buf204  # reuse
    buf216 = buf203; del buf203  # reuse
    buf217 = empty((384, ), device='cpu', dtype=torch.float32)
    buf218 = empty((384, ), device='cpu', dtype=torch.float32)
    buf219 = buf207; del buf207  # reuse
    buf220 = reinterpret_tensor(buf187, (3072, 196), (196, 1), 0); del buf187  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_34(c_void_p(buf219.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(mul_124.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()))
    del div_18
    del mul_124
    del primals_189
    buf221 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf220, permute_320, out=buf221)
    del permute_320
    buf222 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (196, 3072), (1, 196), 0), view_123, out=buf222)
    del view_123
    buf223 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf224 = buf199; del buf199  # reuse
    buf225 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_35(c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(getitem_123.data_ptr()), c_void_p(getitem_122.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()))
    del getitem_122
    del getitem_123
    buf226 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (384, 3072), (1, 384), 0), view_121, out=buf226)
    del view_121
    buf227 = buf220; del buf220  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (3072, 384), (384, 1), 0), permute_327, out=buf227)
    del permute_327
    buf228 = buf216; del buf216  # reuse
    buf229 = buf215; del buf215  # reuse
    buf230 = empty((384, ), device='cpu', dtype=torch.float32)
    buf231 = empty((384, ), device='cpu', dtype=torch.float32)
    buf232 = buf219; del buf219  # reuse
    cpp_fused_add_native_layer_norm_backward_36(c_void_p(buf232.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(mul_120.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del div_19
    del mul_120
    del primals_183
    buf233 = buf208; del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (1568, 384), (384, 1), 0), permute_330, out=buf233)
    del permute_330
    buf234 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (384, 1568), (1, 384), 0), view_119, out=buf234)
    del view_119
    buf235 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf236 = buf211; del buf211  # reuse
    cpp_fused_cat_sum_37(c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(getitem_119.data_ptr()), c_void_p(getitem_118.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    del getitem_118
    del getitem_119
    buf237 = reinterpret_tensor(buf227, (1568, 384), (384, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf236, (1568, 1536), (1536, 1), 0), permute_335, out=buf237)
    del permute_335
    buf238 = reinterpret_tensor(buf221, (1536, 384), (384, 1), 0); del buf221  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf236, (1536, 1568), (1, 1536), 0), view_117, out=buf238)
    del view_117
    buf239 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf240 = buf229; del buf229  # reuse
    buf241 = buf228; del buf228  # reuse
    buf242 = empty((384, ), device='cpu', dtype=torch.float32)
    buf243 = empty((384, ), device='cpu', dtype=torch.float32)
    buf244 = buf232; del buf232  # reuse
    buf245 = reinterpret_tensor(buf212, (3072, 196), (196, 1), 0); del buf212  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_38(c_void_p(buf244.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(mul_116.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()))
    del div_20
    del mul_116
    del primals_177
    buf246 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf245, permute_340, out=buf246)
    del permute_340
    buf247 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (196, 3072), (1, 196), 0), view_115, out=buf247)
    del view_115
    buf248 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf249 = buf224; del buf224  # reuse
    buf250 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_39(c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(getitem_115.data_ptr()), c_void_p(getitem_114.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    del getitem_114
    del getitem_115
    buf251 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (384, 3072), (1, 384), 0), view_113, out=buf251)
    del view_113
    buf252 = buf245; del buf245  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (3072, 384), (384, 1), 0), permute_347, out=buf252)
    del permute_347
    buf253 = buf241; del buf241  # reuse
    buf254 = buf240; del buf240  # reuse
    buf255 = empty((384, ), device='cpu', dtype=torch.float32)
    buf256 = empty((384, ), device='cpu', dtype=torch.float32)
    buf257 = buf244; del buf244  # reuse
    cpp_fused_add_native_layer_norm_backward_40(c_void_p(buf257.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(mul_112.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del div_21
    del mul_112
    del primals_171
    buf258 = buf233; del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (1568, 384), (384, 1), 0), permute_350, out=buf258)
    del permute_350
    buf259 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (384, 1568), (1, 384), 0), view_111, out=buf259)
    del view_111
    buf260 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf261 = buf236; del buf236  # reuse
    cpp_fused_cat_sum_41(c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(getitem_110.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del getitem_110
    del getitem_111
    buf262 = reinterpret_tensor(buf252, (1568, 384), (384, 1), 0); del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (1568, 1536), (1536, 1), 0), permute_355, out=buf262)
    del permute_355
    buf263 = reinterpret_tensor(buf246, (1536, 384), (384, 1), 0); del buf246  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (1536, 1568), (1, 1536), 0), view_109, out=buf263)
    del view_109
    buf264 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf265 = buf254; del buf254  # reuse
    buf266 = buf253; del buf253  # reuse
    buf267 = empty((384, ), device='cpu', dtype=torch.float32)
    buf268 = empty((384, ), device='cpu', dtype=torch.float32)
    buf269 = buf257; del buf257  # reuse
    buf270 = reinterpret_tensor(buf237, (3072, 196), (196, 1), 0); del buf237  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_42(c_void_p(buf269.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(mul_108.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()))
    del div_22
    del mul_108
    del primals_165
    buf271 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf270, permute_360, out=buf271)
    del permute_360
    buf272 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (196, 3072), (1, 196), 0), view_107, out=buf272)
    del view_107
    buf273 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf274 = buf249; del buf249  # reuse
    buf275 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_43(c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(getitem_106.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()))
    del getitem_106
    del getitem_107
    buf276 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (384, 3072), (1, 384), 0), view_105, out=buf276)
    del view_105
    buf277 = buf270; del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (3072, 384), (384, 1), 0), permute_367, out=buf277)
    del permute_367
    buf278 = buf266; del buf266  # reuse
    buf279 = buf265; del buf265  # reuse
    buf280 = empty((384, ), device='cpu', dtype=torch.float32)
    buf281 = empty((384, ), device='cpu', dtype=torch.float32)
    buf282 = buf269; del buf269  # reuse
    cpp_fused_add_native_layer_norm_backward_44(c_void_p(buf282.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(mul_104.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del div_23
    del mul_104
    del primals_159
    buf283 = buf258; del buf258  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (1568, 384), (384, 1), 0), permute_370, out=buf283)
    del permute_370
    buf284 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (384, 1568), (1, 384), 0), view_103, out=buf284)
    del view_103
    buf285 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf286 = buf261; del buf261  # reuse
    cpp_fused_cat_sum_45(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(getitem_103.data_ptr()), c_void_p(getitem_102.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del getitem_102
    del getitem_103
    buf287 = reinterpret_tensor(buf277, (1568, 384), (384, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (1568, 1536), (1536, 1), 0), permute_375, out=buf287)
    del permute_375
    buf288 = reinterpret_tensor(buf271, (1536, 384), (384, 1), 0); del buf271  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (1536, 1568), (1, 1536), 0), view_101, out=buf288)
    del view_101
    buf289 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf290 = buf279; del buf279  # reuse
    buf291 = buf278; del buf278  # reuse
    buf292 = empty((384, ), device='cpu', dtype=torch.float32)
    buf293 = empty((384, ), device='cpu', dtype=torch.float32)
    buf294 = buf282; del buf282  # reuse
    buf295 = reinterpret_tensor(buf262, (3072, 196), (196, 1), 0); del buf262  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_46(c_void_p(buf294.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(mul_100.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()))
    del div_24
    del mul_100
    del primals_153
    buf296 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf295, permute_380, out=buf296)
    del permute_380
    buf297 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (196, 3072), (1, 196), 0), view_99, out=buf297)
    del view_99
    buf298 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf299 = buf274; del buf274  # reuse
    buf300 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_47(c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(getitem_99.data_ptr()), c_void_p(getitem_98.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del getitem_98
    del getitem_99
    buf301 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (384, 3072), (1, 384), 0), view_97, out=buf301)
    del view_97
    buf302 = buf295; del buf295  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (3072, 384), (384, 1), 0), permute_387, out=buf302)
    del permute_387
    buf303 = buf291; del buf291  # reuse
    buf304 = buf290; del buf290  # reuse
    buf305 = empty((384, ), device='cpu', dtype=torch.float32)
    buf306 = empty((384, ), device='cpu', dtype=torch.float32)
    buf307 = buf294; del buf294  # reuse
    cpp_fused_add_native_layer_norm_backward_48(c_void_p(buf307.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del div_25
    del mul_96
    del primals_147
    buf308 = buf283; del buf283  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (1568, 384), (384, 1), 0), permute_390, out=buf308)
    del permute_390
    buf309 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (384, 1568), (1, 384), 0), view_95, out=buf309)
    del view_95
    buf310 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf311 = buf286; del buf286  # reuse
    cpp_fused_cat_sum_49(c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(getitem_95.data_ptr()), c_void_p(getitem_94.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    del getitem_94
    del getitem_95
    buf312 = reinterpret_tensor(buf302, (1568, 384), (384, 1), 0); del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (1568, 1536), (1536, 1), 0), permute_395, out=buf312)
    del permute_395
    buf313 = reinterpret_tensor(buf296, (1536, 384), (384, 1), 0); del buf296  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (1536, 1568), (1, 1536), 0), view_93, out=buf313)
    del view_93
    buf314 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf315 = buf304; del buf304  # reuse
    buf316 = buf303; del buf303  # reuse
    buf317 = empty((384, ), device='cpu', dtype=torch.float32)
    buf318 = empty((384, ), device='cpu', dtype=torch.float32)
    buf319 = buf307; del buf307  # reuse
    buf320 = reinterpret_tensor(buf287, (3072, 196), (196, 1), 0); del buf287  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_50(c_void_p(buf319.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(mul_92.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()))
    del div_26
    del mul_92
    del primals_141
    buf321 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, permute_400, out=buf321)
    del permute_400
    buf322 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf320, (196, 3072), (1, 196), 0), view_91, out=buf322)
    del view_91
    buf323 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf324 = buf299; del buf299  # reuse
    buf325 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_51(c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(getitem_90.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del getitem_90
    del getitem_91
    buf326 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (384, 3072), (1, 384), 0), view_89, out=buf326)
    del view_89
    buf327 = buf320; del buf320  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (3072, 384), (384, 1), 0), permute_407, out=buf327)
    del permute_407
    buf328 = buf316; del buf316  # reuse
    buf329 = buf315; del buf315  # reuse
    buf330 = empty((384, ), device='cpu', dtype=torch.float32)
    buf331 = empty((384, ), device='cpu', dtype=torch.float32)
    buf332 = buf319; del buf319  # reuse
    cpp_fused_add_native_layer_norm_backward_52(c_void_p(buf332.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    del div_27
    del mul_88
    del primals_135
    buf333 = buf308; del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf332, (1568, 384), (384, 1), 0), permute_410, out=buf333)
    del permute_410
    buf334 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf332, (384, 1568), (1, 384), 0), view_87, out=buf334)
    del view_87
    buf335 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf336 = buf311; del buf311  # reuse
    cpp_fused_cat_sum_53(c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(getitem_86.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del getitem_86
    del getitem_87
    buf337 = reinterpret_tensor(buf327, (1568, 384), (384, 1), 0); del buf327  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (1568, 1536), (1536, 1), 0), permute_415, out=buf337)
    del permute_415
    buf338 = reinterpret_tensor(buf321, (1536, 384), (384, 1), 0); del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (1536, 1568), (1, 1536), 0), view_85, out=buf338)
    del view_85
    buf339 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf340 = buf329; del buf329  # reuse
    buf341 = buf328; del buf328  # reuse
    buf342 = empty((384, ), device='cpu', dtype=torch.float32)
    buf343 = empty((384, ), device='cpu', dtype=torch.float32)
    buf344 = buf332; del buf332  # reuse
    buf345 = reinterpret_tensor(buf312, (3072, 196), (196, 1), 0); del buf312  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_54(c_void_p(buf344.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(mul_84.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()))
    del div_28
    del mul_84
    del primals_129
    buf346 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf345, permute_420, out=buf346)
    del permute_420
    buf347 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (196, 3072), (1, 196), 0), view_83, out=buf347)
    del view_83
    buf348 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf349 = buf324; del buf324  # reuse
    buf350 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_55(c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(getitem_83.data_ptr()), c_void_p(getitem_82.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    del getitem_82
    del getitem_83
    buf351 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (384, 3072), (1, 384), 0), view_81, out=buf351)
    del view_81
    buf352 = buf345; del buf345  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (3072, 384), (384, 1), 0), permute_427, out=buf352)
    del permute_427
    buf353 = buf341; del buf341  # reuse
    buf354 = buf340; del buf340  # reuse
    buf355 = empty((384, ), device='cpu', dtype=torch.float32)
    buf356 = empty((384, ), device='cpu', dtype=torch.float32)
    buf357 = buf344; del buf344  # reuse
    cpp_fused_add_native_layer_norm_backward_56(c_void_p(buf357.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    del div_29
    del mul_80
    del primals_123
    buf358 = buf333; del buf333  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf357, (1568, 384), (384, 1), 0), permute_430, out=buf358)
    del permute_430
    buf359 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf357, (384, 1568), (1, 384), 0), view_79, out=buf359)
    del view_79
    buf360 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf361 = buf336; del buf336  # reuse
    cpp_fused_cat_sum_57(c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(getitem_78.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del getitem_78
    del getitem_79
    buf362 = reinterpret_tensor(buf352, (1568, 384), (384, 1), 0); del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (1568, 1536), (1536, 1), 0), permute_435, out=buf362)
    del permute_435
    buf363 = reinterpret_tensor(buf346, (1536, 384), (384, 1), 0); del buf346  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (1536, 1568), (1, 1536), 0), view_77, out=buf363)
    del view_77
    buf364 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf365 = buf354; del buf354  # reuse
    buf366 = buf353; del buf353  # reuse
    buf367 = empty((384, ), device='cpu', dtype=torch.float32)
    buf368 = empty((384, ), device='cpu', dtype=torch.float32)
    buf369 = buf357; del buf357  # reuse
    buf370 = reinterpret_tensor(buf337, (3072, 196), (196, 1), 0); del buf337  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_58(c_void_p(buf369.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(mul_76.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf370.data_ptr()))
    del div_30
    del mul_76
    del primals_117
    buf371 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf370, permute_440, out=buf371)
    del permute_440
    buf372 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf370, (196, 3072), (1, 196), 0), view_75, out=buf372)
    del view_75
    buf373 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf374 = buf349; del buf349  # reuse
    buf375 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_59(c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(getitem_75.data_ptr()), c_void_p(getitem_74.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    del getitem_74
    del getitem_75
    buf376 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (384, 3072), (1, 384), 0), view_73, out=buf376)
    del view_73
    buf377 = buf370; del buf370  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (3072, 384), (384, 1), 0), permute_447, out=buf377)
    del permute_447
    buf378 = buf366; del buf366  # reuse
    buf379 = buf365; del buf365  # reuse
    buf380 = empty((384, ), device='cpu', dtype=torch.float32)
    buf381 = empty((384, ), device='cpu', dtype=torch.float32)
    buf382 = buf369; del buf369  # reuse
    cpp_fused_add_native_layer_norm_backward_60(c_void_p(buf382.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    del div_31
    del mul_72
    del primals_111
    buf383 = buf358; del buf358  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf382, (1568, 384), (384, 1), 0), permute_450, out=buf383)
    del permute_450
    buf384 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf382, (384, 1568), (1, 384), 0), view_71, out=buf384)
    del view_71
    buf385 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf386 = buf361; del buf361  # reuse
    cpp_fused_cat_sum_61(c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(getitem_70.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del getitem_70
    del getitem_71
    buf387 = reinterpret_tensor(buf377, (1568, 384), (384, 1), 0); del buf377  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (1568, 1536), (1536, 1), 0), permute_455, out=buf387)
    del permute_455
    buf388 = reinterpret_tensor(buf371, (1536, 384), (384, 1), 0); del buf371  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (1536, 1568), (1, 1536), 0), view_69, out=buf388)
    del view_69
    buf389 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf390 = buf379; del buf379  # reuse
    buf391 = buf378; del buf378  # reuse
    buf392 = empty((384, ), device='cpu', dtype=torch.float32)
    buf393 = empty((384, ), device='cpu', dtype=torch.float32)
    buf394 = buf382; del buf382  # reuse
    buf395 = reinterpret_tensor(buf362, (3072, 196), (196, 1), 0); del buf362  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_62(c_void_p(buf394.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(mul_68.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf395.data_ptr()))
    del div_32
    del mul_68
    del primals_105
    buf396 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf395, permute_460, out=buf396)
    del permute_460
    buf397 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (196, 3072), (1, 196), 0), view_67, out=buf397)
    del view_67
    buf398 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf399 = buf374; del buf374  # reuse
    buf400 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_63(c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(getitem_66.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()))
    del getitem_66
    del getitem_67
    buf401 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (384, 3072), (1, 384), 0), view_65, out=buf401)
    del view_65
    buf402 = buf395; del buf395  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (3072, 384), (384, 1), 0), permute_467, out=buf402)
    del permute_467
    buf403 = buf391; del buf391  # reuse
    buf404 = buf390; del buf390  # reuse
    buf405 = empty((384, ), device='cpu', dtype=torch.float32)
    buf406 = empty((384, ), device='cpu', dtype=torch.float32)
    buf407 = buf394; del buf394  # reuse
    cpp_fused_add_native_layer_norm_backward_64(c_void_p(buf407.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()))
    del div_33
    del mul_64
    del primals_99
    buf408 = buf383; del buf383  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf407, (1568, 384), (384, 1), 0), permute_470, out=buf408)
    del permute_470
    buf409 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf407, (384, 1568), (1, 384), 0), view_63, out=buf409)
    del view_63
    buf410 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf411 = buf386; del buf386  # reuse
    cpp_fused_cat_sum_65(c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(getitem_63.data_ptr()), c_void_p(getitem_62.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    del getitem_62
    del getitem_63
    buf412 = reinterpret_tensor(buf402, (1568, 384), (384, 1), 0); del buf402  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf411, (1568, 1536), (1536, 1), 0), permute_475, out=buf412)
    del permute_475
    buf413 = reinterpret_tensor(buf396, (1536, 384), (384, 1), 0); del buf396  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf411, (1536, 1568), (1, 1536), 0), view_61, out=buf413)
    del view_61
    buf414 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf415 = buf404; del buf404  # reuse
    buf416 = buf403; del buf403  # reuse
    buf417 = empty((384, ), device='cpu', dtype=torch.float32)
    buf418 = empty((384, ), device='cpu', dtype=torch.float32)
    buf419 = buf407; del buf407  # reuse
    buf420 = reinterpret_tensor(buf387, (3072, 196), (196, 1), 0); del buf387  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_66(c_void_p(buf419.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(mul_60.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf420.data_ptr()))
    del div_34
    del mul_60
    del primals_93
    buf421 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf420, permute_480, out=buf421)
    del permute_480
    buf422 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf420, (196, 3072), (1, 196), 0), view_59, out=buf422)
    del view_59
    buf423 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf424 = buf399; del buf399  # reuse
    buf425 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_67(c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(getitem_59.data_ptr()), c_void_p(getitem_58.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()))
    del getitem_58
    del getitem_59
    buf426 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf424, (384, 3072), (1, 384), 0), view_57, out=buf426)
    del view_57
    buf427 = buf420; del buf420  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf424, (3072, 384), (384, 1), 0), permute_487, out=buf427)
    del permute_487
    buf428 = buf416; del buf416  # reuse
    buf429 = buf415; del buf415  # reuse
    buf430 = empty((384, ), device='cpu', dtype=torch.float32)
    buf431 = empty((384, ), device='cpu', dtype=torch.float32)
    buf432 = buf419; del buf419  # reuse
    cpp_fused_add_native_layer_norm_backward_68(c_void_p(buf432.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()))
    del div_35
    del mul_56
    del primals_87
    buf433 = buf408; del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (1568, 384), (384, 1), 0), permute_490, out=buf433)
    del permute_490
    buf434 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (384, 1568), (1, 384), 0), view_55, out=buf434)
    del view_55
    buf435 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf436 = buf411; del buf411  # reuse
    cpp_fused_cat_sum_69(c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(getitem_55.data_ptr()), c_void_p(getitem_54.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del getitem_54
    del getitem_55
    buf437 = reinterpret_tensor(buf427, (1568, 384), (384, 1), 0); del buf427  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (1568, 1536), (1536, 1), 0), permute_495, out=buf437)
    del permute_495
    buf438 = reinterpret_tensor(buf421, (1536, 384), (384, 1), 0); del buf421  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (1536, 1568), (1, 1536), 0), view_53, out=buf438)
    del view_53
    buf439 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf440 = buf429; del buf429  # reuse
    buf441 = buf428; del buf428  # reuse
    buf442 = empty((384, ), device='cpu', dtype=torch.float32)
    buf443 = empty((384, ), device='cpu', dtype=torch.float32)
    buf444 = buf432; del buf432  # reuse
    buf445 = reinterpret_tensor(buf412, (3072, 196), (196, 1), 0); del buf412  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_70(c_void_p(buf444.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(mul_52.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf445.data_ptr()))
    del div_36
    del mul_52
    del primals_81
    buf446 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf445, permute_500, out=buf446)
    del permute_500
    buf447 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf445, (196, 3072), (1, 196), 0), view_51, out=buf447)
    del view_51
    buf448 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf449 = buf424; del buf424  # reuse
    buf450 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_71(c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(getitem_50.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()))
    del getitem_50
    del getitem_51
    buf451 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf449, (384, 3072), (1, 384), 0), view_49, out=buf451)
    del view_49
    buf452 = buf445; del buf445  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf449, (3072, 384), (384, 1), 0), permute_507, out=buf452)
    del permute_507
    buf453 = buf441; del buf441  # reuse
    buf454 = buf440; del buf440  # reuse
    buf455 = empty((384, ), device='cpu', dtype=torch.float32)
    buf456 = empty((384, ), device='cpu', dtype=torch.float32)
    buf457 = buf444; del buf444  # reuse
    cpp_fused_add_native_layer_norm_backward_72(c_void_p(buf457.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    del div_37
    del mul_48
    del primals_75
    buf458 = buf433; del buf433  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf457, (1568, 384), (384, 1), 0), permute_510, out=buf458)
    del permute_510
    buf459 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf457, (384, 1568), (1, 384), 0), view_47, out=buf459)
    del view_47
    buf460 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf461 = buf436; del buf436  # reuse
    cpp_fused_cat_sum_73(c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(getitem_46.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()))
    del getitem_46
    del getitem_47
    buf462 = reinterpret_tensor(buf452, (1568, 384), (384, 1), 0); del buf452  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (1568, 1536), (1536, 1), 0), permute_515, out=buf462)
    del permute_515
    buf463 = reinterpret_tensor(buf446, (1536, 384), (384, 1), 0); del buf446  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (1536, 1568), (1, 1536), 0), view_45, out=buf463)
    del view_45
    buf464 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf465 = buf454; del buf454  # reuse
    buf466 = buf453; del buf453  # reuse
    buf467 = empty((384, ), device='cpu', dtype=torch.float32)
    buf468 = empty((384, ), device='cpu', dtype=torch.float32)
    buf469 = buf457; del buf457  # reuse
    buf470 = reinterpret_tensor(buf437, (3072, 196), (196, 1), 0); del buf437  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_74(c_void_p(buf469.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(mul_44.data_ptr()), c_void_p(div_38.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()))
    del div_38
    del mul_44
    del primals_69
    buf471 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf470, permute_520, out=buf471)
    del permute_520
    buf472 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf470, (196, 3072), (1, 196), 0), view_43, out=buf472)
    del view_43
    buf473 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf474 = buf449; del buf449  # reuse
    buf475 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_75(c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(getitem_43.data_ptr()), c_void_p(getitem_42.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()))
    del getitem_42
    del getitem_43
    buf476 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (384, 3072), (1, 384), 0), view_41, out=buf476)
    del view_41
    buf477 = buf470; del buf470  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (3072, 384), (384, 1), 0), permute_527, out=buf477)
    del permute_527
    buf478 = buf466; del buf466  # reuse
    buf479 = buf465; del buf465  # reuse
    buf480 = empty((384, ), device='cpu', dtype=torch.float32)
    buf481 = empty((384, ), device='cpu', dtype=torch.float32)
    buf482 = buf469; del buf469  # reuse
    cpp_fused_add_native_layer_norm_backward_76(c_void_p(buf482.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()))
    del div_39
    del mul_40
    del primals_63
    buf483 = buf458; del buf458  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (1568, 384), (384, 1), 0), permute_530, out=buf483)
    del permute_530
    buf484 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (384, 1568), (1, 384), 0), view_39, out=buf484)
    del view_39
    buf485 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf486 = buf461; del buf461  # reuse
    cpp_fused_cat_sum_77(c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(getitem_39.data_ptr()), c_void_p(getitem_38.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    del getitem_38
    del getitem_39
    buf487 = reinterpret_tensor(buf477, (1568, 384), (384, 1), 0); del buf477  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (1568, 1536), (1536, 1), 0), permute_535, out=buf487)
    del permute_535
    buf488 = reinterpret_tensor(buf471, (1536, 384), (384, 1), 0); del buf471  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (1536, 1568), (1, 1536), 0), view_37, out=buf488)
    del view_37
    buf489 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf490 = buf479; del buf479  # reuse
    buf491 = buf478; del buf478  # reuse
    buf492 = empty((384, ), device='cpu', dtype=torch.float32)
    buf493 = empty((384, ), device='cpu', dtype=torch.float32)
    buf494 = buf482; del buf482  # reuse
    buf495 = reinterpret_tensor(buf462, (3072, 196), (196, 1), 0); del buf462  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_78(c_void_p(buf494.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(mul_36.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf495.data_ptr()))
    del div_40
    del mul_36
    del primals_57
    buf496 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf495, permute_540, out=buf496)
    del permute_540
    buf497 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (196, 3072), (1, 196), 0), view_35, out=buf497)
    del view_35
    buf498 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf499 = buf474; del buf474  # reuse
    buf500 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_79(c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(getitem_35.data_ptr()), c_void_p(getitem_34.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del getitem_34
    del getitem_35
    buf501 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf499, (384, 3072), (1, 384), 0), view_33, out=buf501)
    del view_33
    buf502 = buf495; del buf495  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf499, (3072, 384), (384, 1), 0), permute_547, out=buf502)
    del permute_547
    buf503 = buf491; del buf491  # reuse
    buf504 = buf490; del buf490  # reuse
    buf505 = empty((384, ), device='cpu', dtype=torch.float32)
    buf506 = empty((384, ), device='cpu', dtype=torch.float32)
    buf507 = buf494; del buf494  # reuse
    cpp_fused_add_native_layer_norm_backward_80(c_void_p(buf507.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()))
    del div_41
    del mul_32
    del primals_51
    buf508 = buf483; del buf483  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf507, (1568, 384), (384, 1), 0), permute_550, out=buf508)
    del permute_550
    buf509 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf507, (384, 1568), (1, 384), 0), view_31, out=buf509)
    del view_31
    buf510 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf511 = buf486; del buf486  # reuse
    cpp_fused_cat_sum_81(c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(getitem_30.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()))
    del getitem_30
    del getitem_31
    buf512 = reinterpret_tensor(buf502, (1568, 384), (384, 1), 0); del buf502  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf511, (1568, 1536), (1536, 1), 0), permute_555, out=buf512)
    del permute_555
    buf513 = reinterpret_tensor(buf496, (1536, 384), (384, 1), 0); del buf496  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf511, (1536, 1568), (1, 1536), 0), view_29, out=buf513)
    del view_29
    buf514 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf515 = buf504; del buf504  # reuse
    buf516 = buf503; del buf503  # reuse
    buf517 = empty((384, ), device='cpu', dtype=torch.float32)
    buf518 = empty((384, ), device='cpu', dtype=torch.float32)
    buf519 = buf507; del buf507  # reuse
    buf520 = reinterpret_tensor(buf487, (3072, 196), (196, 1), 0); del buf487  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_82(c_void_p(buf519.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf520.data_ptr()))
    del div_42
    del mul_28
    del primals_45
    buf521 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf520, permute_560, out=buf521)
    del permute_560
    buf522 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf520, (196, 3072), (1, 196), 0), view_27, out=buf522)
    del view_27
    buf523 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf524 = buf499; del buf499  # reuse
    buf525 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_83(c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(getitem_26.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf525.data_ptr()))
    del getitem_26
    del getitem_27
    buf526 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (384, 3072), (1, 384), 0), view_25, out=buf526)
    del view_25
    buf527 = buf520; del buf520  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (3072, 384), (384, 1), 0), permute_567, out=buf527)
    del permute_567
    buf528 = buf516; del buf516  # reuse
    buf529 = buf515; del buf515  # reuse
    buf530 = empty((384, ), device='cpu', dtype=torch.float32)
    buf531 = empty((384, ), device='cpu', dtype=torch.float32)
    buf532 = buf519; del buf519  # reuse
    cpp_fused_add_native_layer_norm_backward_84(c_void_p(buf532.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()))
    del div_43
    del mul_24
    del primals_39
    buf533 = buf508; del buf508  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf532, (1568, 384), (384, 1), 0), permute_570, out=buf533)
    del permute_570
    buf534 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf532, (384, 1568), (1, 384), 0), view_23, out=buf534)
    del view_23
    buf535 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf536 = buf511; del buf511  # reuse
    cpp_fused_cat_sum_85(c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(getitem_23.data_ptr()), c_void_p(getitem_22.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()))
    del getitem_22
    del getitem_23
    buf537 = reinterpret_tensor(buf527, (1568, 384), (384, 1), 0); del buf527  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf536, (1568, 1536), (1536, 1), 0), permute_575, out=buf537)
    del permute_575
    buf538 = reinterpret_tensor(buf521, (1536, 384), (384, 1), 0); del buf521  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf536, (1536, 1568), (1, 1536), 0), view_21, out=buf538)
    del view_21
    buf539 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf540 = buf529; del buf529  # reuse
    buf541 = buf528; del buf528  # reuse
    buf542 = empty((384, ), device='cpu', dtype=torch.float32)
    buf543 = empty((384, ), device='cpu', dtype=torch.float32)
    buf544 = buf532; del buf532  # reuse
    buf545 = reinterpret_tensor(buf512, (3072, 196), (196, 1), 0); del buf512  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_86(c_void_p(buf544.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(mul_20.data_ptr()), c_void_p(div_44.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf545.data_ptr()))
    del div_44
    del mul_20
    del primals_33
    buf546 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf545, permute_580, out=buf546)
    del permute_580
    buf547 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf545, (196, 3072), (1, 196), 0), view_19, out=buf547)
    del view_19
    buf548 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf549 = buf524; del buf524  # reuse
    buf550 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_87(c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(getitem_19.data_ptr()), c_void_p(getitem_18.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()))
    del getitem_18
    del getitem_19
    buf551 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf549, (384, 3072), (1, 384), 0), view_17, out=buf551)
    del view_17
    buf552 = buf545; del buf545  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf549, (3072, 384), (384, 1), 0), permute_587, out=buf552)
    del permute_587
    buf553 = buf541; del buf541  # reuse
    buf554 = buf540; del buf540  # reuse
    buf555 = empty((384, ), device='cpu', dtype=torch.float32)
    buf556 = empty((384, ), device='cpu', dtype=torch.float32)
    buf557 = buf544; del buf544  # reuse
    cpp_fused_add_native_layer_norm_backward_88(c_void_p(buf557.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()))
    del div_45
    del mul_16
    del primals_27
    buf558 = buf533; del buf533  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf557, (1568, 384), (384, 1), 0), permute_590, out=buf558)
    del permute_590
    buf559 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf557, (384, 1568), (1, 384), 0), view_15, out=buf559)
    del view_15
    buf560 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf561 = buf536; del buf536  # reuse
    cpp_fused_cat_sum_89(c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(getitem_15.data_ptr()), c_void_p(getitem_14.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()))
    del getitem_14
    del getitem_15
    buf562 = reinterpret_tensor(buf552, (1568, 384), (384, 1), 0); del buf552  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf561, (1568, 1536), (1536, 1), 0), permute_595, out=buf562)
    del permute_595
    buf563 = reinterpret_tensor(buf546, (1536, 384), (384, 1), 0); del buf546  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf561, (1536, 1568), (1, 1536), 0), view_13, out=buf563)
    del view_13
    buf564 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf565 = buf554; del buf554  # reuse
    buf566 = buf553; del buf553  # reuse
    buf567 = empty((384, ), device='cpu', dtype=torch.float32)
    buf568 = empty((384, ), device='cpu', dtype=torch.float32)
    buf569 = buf557; del buf557  # reuse
    buf570 = reinterpret_tensor(buf537, (3072, 196), (196, 1), 0); del buf537  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_90(c_void_p(buf569.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(mul_12.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf570.data_ptr()))
    del div_46
    del mul_12
    del primals_21
    buf571 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf570, permute_600, out=buf571)
    del permute_600
    buf572 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf570, (196, 3072), (1, 196), 0), view_11, out=buf572)
    del view_11
    buf573 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf574 = buf549; del buf549  # reuse
    buf575 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_91(c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(getitem_10.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf575.data_ptr()))
    del getitem_10
    del getitem_11
    buf576 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf574, (384, 3072), (1, 384), 0), view_9, out=buf576)
    del view_9
    buf577 = buf570; del buf570  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf574, (3072, 384), (384, 1), 0), permute_607, out=buf577)
    del permute_607
    buf578 = buf566; del buf566  # reuse
    buf579 = buf565; del buf565  # reuse
    buf580 = empty((384, ), device='cpu', dtype=torch.float32)
    buf581 = empty((384, ), device='cpu', dtype=torch.float32)
    buf582 = buf569; del buf569  # reuse
    cpp_fused_add_native_layer_norm_backward_92(c_void_p(buf582.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_47.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()))
    del div_47
    del mul_8
    del primals_15
    buf583 = buf558; del buf558  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf582, (1568, 384), (384, 1), 0), permute_610, out=buf583)
    del permute_610
    buf584 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf582, (384, 1568), (1, 384), 0), view_7, out=buf584)
    del view_7
    buf585 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf586 = buf561; del buf561  # reuse
    cpp_fused_cat_sum_93(c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(getitem_6.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()))
    del buf583
    del getitem_6
    del getitem_7
    buf587 = reinterpret_tensor(buf577, (1568, 384), (384, 1), 0); del buf577  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf586, (1568, 1536), (1536, 1), 0), permute_615, out=buf587)
    del permute_615
    buf588 = reinterpret_tensor(buf571, (1536, 384), (384, 1), 0); del buf571  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf586, (1536, 1568), (1, 1536), 0), view_5, out=buf588)
    del view_5
    buf589 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf590 = buf579; del buf579  # reuse
    buf591 = buf578; del buf578  # reuse
    buf592 = empty((384, ), device='cpu', dtype=torch.float32)
    buf593 = empty((384, ), device='cpu', dtype=torch.float32)
    buf594 = buf582; del buf582  # reuse
    buf595 = reinterpret_tensor(buf562, (3072, 196), (196, 1), 0); del buf562  # reuse
    cpp_fused__unsafe_view_add_clone_native_layer_norm_backward_sum_94(c_void_p(buf594.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(mul_4.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf595.data_ptr()))
    del buf586
    del buf587
    del div_48
    del mul_4
    del primals_9
    buf596 = empty((3072, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf595, permute_620, out=buf596)
    del permute_620
    buf597 = empty((196, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf595, (196, 3072), (1, 196), 0), view_3, out=buf597)
    del view_3
    buf598 = empty((1, 196), device='cpu', dtype=torch.float32)
    buf599 = buf574; del buf574  # reuse
    buf600 = empty((1, 1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_sum_95(c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(getitem_2.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()))
    del buf596
    del getitem_2
    del getitem_3
    buf601 = empty((384, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf599, (384, 3072), (1, 384), 0), view_1, out=buf601)
    del view_1
    buf602 = buf595; del buf595  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf599, (3072, 384), (384, 1), 0), permute_627, out=buf602)
    del buf599
    del permute_627
    buf603 = buf591; del buf591  # reuse
    buf604 = buf590; del buf590  # reuse
    buf605 = empty((384, ), device='cpu', dtype=torch.float32)
    buf606 = empty((384, ), device='cpu', dtype=torch.float32)
    buf607 = reinterpret_tensor(buf594, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf594  # reuse
    cpp_fused_convolution_backward_native_layer_norm_backward_96(c_void_p(buf607.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    del buf602
    del buf603
    del buf604
    del div_49
    del mul
    del primals_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf608 = aten.convolution_backward(buf607, primals_295, primals_1, [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf607
    del primals_1
    del primals_295
    buf609 = buf608[1]
    buf610 = buf608[2]
    return (buf609, buf610, buf605, buf606, reinterpret_tensor(buf601, (384, 196), (196, 1), 0), reinterpret_tensor(buf600, (384, ), (1, ), 0), reinterpret_tensor(buf597, (196, 192), (192, 1), 0), reinterpret_tensor(buf598, (196, ), (1, ), 0), buf592, buf593, reinterpret_tensor(buf588, (1536, 384), (384, 1), 0), reinterpret_tensor(buf589, (1536, ), (1, ), 0), reinterpret_tensor(buf584, (384, 768), (768, 1), 0), reinterpret_tensor(buf585, (384, ), (1, ), 0), buf580, buf581, reinterpret_tensor(buf576, (384, 196), (196, 1), 0), reinterpret_tensor(buf575, (384, ), (1, ), 0), reinterpret_tensor(buf572, (196, 192), (192, 1), 0), reinterpret_tensor(buf573, (196, ), (1, ), 0), buf567, buf568, reinterpret_tensor(buf563, (1536, 384), (384, 1), 0), reinterpret_tensor(buf564, (1536, ), (1, ), 0), reinterpret_tensor(buf559, (384, 768), (768, 1), 0), reinterpret_tensor(buf560, (384, ), (1, ), 0), buf555, buf556, reinterpret_tensor(buf551, (384, 196), (196, 1), 0), reinterpret_tensor(buf550, (384, ), (1, ), 0), reinterpret_tensor(buf547, (196, 192), (192, 1), 0), reinterpret_tensor(buf548, (196, ), (1, ), 0), buf542, buf543, reinterpret_tensor(buf538, (1536, 384), (384, 1), 0), reinterpret_tensor(buf539, (1536, ), (1, ), 0), reinterpret_tensor(buf534, (384, 768), (768, 1), 0), reinterpret_tensor(buf535, (384, ), (1, ), 0), buf530, buf531, reinterpret_tensor(buf526, (384, 196), (196, 1), 0), reinterpret_tensor(buf525, (384, ), (1, ), 0), reinterpret_tensor(buf522, (196, 192), (192, 1), 0), reinterpret_tensor(buf523, (196, ), (1, ), 0), buf517, buf518, reinterpret_tensor(buf513, (1536, 384), (384, 1), 0), reinterpret_tensor(buf514, (1536, ), (1, ), 0), reinterpret_tensor(buf509, (384, 768), (768, 1), 0), reinterpret_tensor(buf510, (384, ), (1, ), 0), buf505, buf506, reinterpret_tensor(buf501, (384, 196), (196, 1), 0), reinterpret_tensor(buf500, (384, ), (1, ), 0), reinterpret_tensor(buf497, (196, 192), (192, 1), 0), reinterpret_tensor(buf498, (196, ), (1, ), 0), buf492, buf493, reinterpret_tensor(buf488, (1536, 384), (384, 1), 0), reinterpret_tensor(buf489, (1536, ), (1, ), 0), reinterpret_tensor(buf484, (384, 768), (768, 1), 0), reinterpret_tensor(buf485, (384, ), (1, ), 0), buf480, buf481, reinterpret_tensor(buf476, (384, 196), (196, 1), 0), reinterpret_tensor(buf475, (384, ), (1, ), 0), reinterpret_tensor(buf472, (196, 192), (192, 1), 0), reinterpret_tensor(buf473, (196, ), (1, ), 0), buf467, buf468, reinterpret_tensor(buf463, (1536, 384), (384, 1), 0), reinterpret_tensor(buf464, (1536, ), (1, ), 0), reinterpret_tensor(buf459, (384, 768), (768, 1), 0), reinterpret_tensor(buf460, (384, ), (1, ), 0), buf455, buf456, reinterpret_tensor(buf451, (384, 196), (196, 1), 0), reinterpret_tensor(buf450, (384, ), (1, ), 0), reinterpret_tensor(buf447, (196, 192), (192, 1), 0), reinterpret_tensor(buf448, (196, ), (1, ), 0), buf442, buf443, reinterpret_tensor(buf438, (1536, 384), (384, 1), 0), reinterpret_tensor(buf439, (1536, ), (1, ), 0), reinterpret_tensor(buf434, (384, 768), (768, 1), 0), reinterpret_tensor(buf435, (384, ), (1, ), 0), buf430, buf431, reinterpret_tensor(buf426, (384, 196), (196, 1), 0), reinterpret_tensor(buf425, (384, ), (1, ), 0), reinterpret_tensor(buf422, (196, 192), (192, 1), 0), reinterpret_tensor(buf423, (196, ), (1, ), 0), buf417, buf418, reinterpret_tensor(buf413, (1536, 384), (384, 1), 0), reinterpret_tensor(buf414, (1536, ), (1, ), 0), reinterpret_tensor(buf409, (384, 768), (768, 1), 0), reinterpret_tensor(buf410, (384, ), (1, ), 0), buf405, buf406, reinterpret_tensor(buf401, (384, 196), (196, 1), 0), reinterpret_tensor(buf400, (384, ), (1, ), 0), reinterpret_tensor(buf397, (196, 192), (192, 1), 0), reinterpret_tensor(buf398, (196, ), (1, ), 0), buf392, buf393, reinterpret_tensor(buf388, (1536, 384), (384, 1), 0), reinterpret_tensor(buf389, (1536, ), (1, ), 0), reinterpret_tensor(buf384, (384, 768), (768, 1), 0), reinterpret_tensor(buf385, (384, ), (1, ), 0), buf380, buf381, reinterpret_tensor(buf376, (384, 196), (196, 1), 0), reinterpret_tensor(buf375, (384, ), (1, ), 0), reinterpret_tensor(buf372, (196, 192), (192, 1), 0), reinterpret_tensor(buf373, (196, ), (1, ), 0), buf367, buf368, reinterpret_tensor(buf363, (1536, 384), (384, 1), 0), reinterpret_tensor(buf364, (1536, ), (1, ), 0), reinterpret_tensor(buf359, (384, 768), (768, 1), 0), reinterpret_tensor(buf360, (384, ), (1, ), 0), buf355, buf356, reinterpret_tensor(buf351, (384, 196), (196, 1), 0), reinterpret_tensor(buf350, (384, ), (1, ), 0), reinterpret_tensor(buf347, (196, 192), (192, 1), 0), reinterpret_tensor(buf348, (196, ), (1, ), 0), buf342, buf343, reinterpret_tensor(buf338, (1536, 384), (384, 1), 0), reinterpret_tensor(buf339, (1536, ), (1, ), 0), reinterpret_tensor(buf334, (384, 768), (768, 1), 0), reinterpret_tensor(buf335, (384, ), (1, ), 0), buf330, buf331, reinterpret_tensor(buf326, (384, 196), (196, 1), 0), reinterpret_tensor(buf325, (384, ), (1, ), 0), reinterpret_tensor(buf322, (196, 192), (192, 1), 0), reinterpret_tensor(buf323, (196, ), (1, ), 0), buf317, buf318, reinterpret_tensor(buf313, (1536, 384), (384, 1), 0), reinterpret_tensor(buf314, (1536, ), (1, ), 0), reinterpret_tensor(buf309, (384, 768), (768, 1), 0), reinterpret_tensor(buf310, (384, ), (1, ), 0), buf305, buf306, reinterpret_tensor(buf301, (384, 196), (196, 1), 0), reinterpret_tensor(buf300, (384, ), (1, ), 0), reinterpret_tensor(buf297, (196, 192), (192, 1), 0), reinterpret_tensor(buf298, (196, ), (1, ), 0), buf292, buf293, reinterpret_tensor(buf288, (1536, 384), (384, 1), 0), reinterpret_tensor(buf289, (1536, ), (1, ), 0), reinterpret_tensor(buf284, (384, 768), (768, 1), 0), reinterpret_tensor(buf285, (384, ), (1, ), 0), buf280, buf281, reinterpret_tensor(buf276, (384, 196), (196, 1), 0), reinterpret_tensor(buf275, (384, ), (1, ), 0), reinterpret_tensor(buf272, (196, 192), (192, 1), 0), reinterpret_tensor(buf273, (196, ), (1, ), 0), buf267, buf268, reinterpret_tensor(buf263, (1536, 384), (384, 1), 0), reinterpret_tensor(buf264, (1536, ), (1, ), 0), reinterpret_tensor(buf259, (384, 768), (768, 1), 0), reinterpret_tensor(buf260, (384, ), (1, ), 0), buf255, buf256, reinterpret_tensor(buf251, (384, 196), (196, 1), 0), reinterpret_tensor(buf250, (384, ), (1, ), 0), reinterpret_tensor(buf247, (196, 192), (192, 1), 0), reinterpret_tensor(buf248, (196, ), (1, ), 0), buf242, buf243, reinterpret_tensor(buf238, (1536, 384), (384, 1), 0), reinterpret_tensor(buf239, (1536, ), (1, ), 0), reinterpret_tensor(buf234, (384, 768), (768, 1), 0), reinterpret_tensor(buf235, (384, ), (1, ), 0), buf230, buf231, reinterpret_tensor(buf226, (384, 196), (196, 1), 0), reinterpret_tensor(buf225, (384, ), (1, ), 0), reinterpret_tensor(buf222, (196, 192), (192, 1), 0), reinterpret_tensor(buf223, (196, ), (1, ), 0), buf217, buf218, reinterpret_tensor(buf213, (1536, 384), (384, 1), 0), reinterpret_tensor(buf214, (1536, ), (1, ), 0), reinterpret_tensor(buf209, (384, 768), (768, 1), 0), reinterpret_tensor(buf210, (384, ), (1, ), 0), buf205, buf206, reinterpret_tensor(buf201, (384, 196), (196, 1), 0), reinterpret_tensor(buf200, (384, ), (1, ), 0), reinterpret_tensor(buf197, (196, 192), (192, 1), 0), reinterpret_tensor(buf198, (196, ), (1, ), 0), buf192, buf193, reinterpret_tensor(buf188, (1536, 384), (384, 1), 0), reinterpret_tensor(buf189, (1536, ), (1, ), 0), reinterpret_tensor(buf184, (384, 768), (768, 1), 0), reinterpret_tensor(buf185, (384, ), (1, ), 0), buf180, buf181, reinterpret_tensor(buf176, (384, 196), (196, 1), 0), reinterpret_tensor(buf175, (384, ), (1, ), 0), reinterpret_tensor(buf172, (196, 192), (192, 1), 0), reinterpret_tensor(buf173, (196, ), (1, ), 0), buf167, buf168, reinterpret_tensor(buf163, (1536, 384), (384, 1), 0), reinterpret_tensor(buf164, (1536, ), (1, ), 0), reinterpret_tensor(buf159, (384, 768), (768, 1), 0), reinterpret_tensor(buf160, (384, ), (1, ), 0), buf155, buf156, reinterpret_tensor(buf151, (384, 196), (196, 1), 0), reinterpret_tensor(buf150, (384, ), (1, ), 0), reinterpret_tensor(buf147, (196, 192), (192, 1), 0), reinterpret_tensor(buf148, (196, ), (1, ), 0), buf142, buf143, reinterpret_tensor(buf138, (1536, 384), (384, 1), 0), reinterpret_tensor(buf139, (1536, ), (1, ), 0), reinterpret_tensor(buf134, (384, 768), (768, 1), 0), reinterpret_tensor(buf135, (384, ), (1, ), 0), buf130, buf131, reinterpret_tensor(buf126, (384, 196), (196, 1), 0), reinterpret_tensor(buf125, (384, ), (1, ), 0), reinterpret_tensor(buf122, (196, 192), (192, 1), 0), reinterpret_tensor(buf123, (196, ), (1, ), 0), buf117, buf118, reinterpret_tensor(buf113, (1536, 384), (384, 1), 0), reinterpret_tensor(buf114, (1536, ), (1, ), 0), reinterpret_tensor(buf109, (384, 768), (768, 1), 0), reinterpret_tensor(buf110, (384, ), (1, ), 0), buf105, buf106, reinterpret_tensor(buf101, (384, 196), (196, 1), 0), reinterpret_tensor(buf100, (384, ), (1, ), 0), reinterpret_tensor(buf97, (196, 192), (192, 1), 0), reinterpret_tensor(buf98, (196, ), (1, ), 0), buf92, buf93, reinterpret_tensor(buf88, (1536, 384), (384, 1), 0), reinterpret_tensor(buf89, (1536, ), (1, ), 0), reinterpret_tensor(buf84, (384, 768), (768, 1), 0), reinterpret_tensor(buf85, (384, ), (1, ), 0), buf80, buf81, reinterpret_tensor(buf76, (384, 196), (196, 1), 0), reinterpret_tensor(buf75, (384, ), (1, ), 0), reinterpret_tensor(buf72, (196, 192), (192, 1), 0), reinterpret_tensor(buf73, (196, ), (1, ), 0), buf67, buf68, reinterpret_tensor(buf63, (1536, 384), (384, 1), 0), reinterpret_tensor(buf64, (1536, ), (1, ), 0), reinterpret_tensor(buf59, (384, 768), (768, 1), 0), reinterpret_tensor(buf60, (384, ), (1, ), 0), buf55, buf56, reinterpret_tensor(buf51, (384, 196), (196, 1), 0), reinterpret_tensor(buf50, (384, ), (1, ), 0), reinterpret_tensor(buf47, (196, 192), (192, 1), 0), reinterpret_tensor(buf48, (196, ), (1, ), 0), buf42, buf43, reinterpret_tensor(buf38, (1536, 384), (384, 1), 0), reinterpret_tensor(buf39, (1536, ), (1, ), 0), reinterpret_tensor(buf34, (384, 768), (768, 1), 0), reinterpret_tensor(buf35, (384, ), (1, ), 0), buf30, buf31, reinterpret_tensor(buf26, (384, 196), (196, 1), 0), reinterpret_tensor(buf25, (384, ), (1, ), 0), reinterpret_tensor(buf22, (196, 192), (192, 1), 0), reinterpret_tensor(buf23, (196, ), (1, ), 0), buf17, buf18, reinterpret_tensor(buf13, (1536, 384), (384, 1), 0), reinterpret_tensor(buf14, (1536, ), (1, ), 0), reinterpret_tensor(buf9, (384, 768), (768, 1), 0), reinterpret_tensor(buf10, (384, ), (1, ), 0), buf6, buf7, reinterpret_tensor(buf1, (1000, 384), (384, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_3 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_4 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_8 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_10 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_12 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_14 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_15 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_18 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_20 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_22 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_24 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_26 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_28 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_30 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_32 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_33 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_34 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_36 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_38 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_42 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_43 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_43 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_44 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_46 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_48 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_50 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_52 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_54 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_55 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_58 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_59 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_60 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_62 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_63 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_63 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_64 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_66 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_68 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_70 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_74 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_75 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_76 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_78 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_79 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_82 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_84 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_86 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_87 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_90 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_92 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_94 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_95 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_96 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_97 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_98 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_99 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_100 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_102 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_103 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_103 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_104 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_105 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_106 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_108 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_109 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_110 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_111 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_112 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_113 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_114 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_115 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_116 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_118 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_119 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_121 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_122 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_123 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_123 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_124 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_126 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_127 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_128 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_129 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_130 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_131 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_132 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_133 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_134 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_135 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_136 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_137 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_138 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_139 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_139 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_140 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_141 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_142 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_143 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_143 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_144 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_145 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_146 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_147 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_148 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_149 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_150 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_151 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_151 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_152 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_153 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_154 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_155 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_155 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_156 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_157 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_158 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_159 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_159 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_160 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_161 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_162 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_163 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_163 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_164 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_165 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_166 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_167 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_167 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_168 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_169 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_170 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_171 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_171 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_172 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_173 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_174 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_175 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_175 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_176 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_177 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_178 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_179 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_179 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_180 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_181 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_182 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_183 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_183 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_184 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_185 = rand_strided((3072, 196), (196, 1), device='cpu', dtype=torch.float32)
    getitem_186 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    getitem_187 = rand_strided((8, 384, 192), (147456, 384, 1), device='cpu', dtype=torch.float32)
    view_187 = rand_strided((3072, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_188 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    view_189 = rand_strided((1568, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_190 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    getitem_191 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    view_191 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((8, 196, 384), (75264, 384, 1), device='cpu', dtype=torch.float32)
    clone_169 = rand_strided((8, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_160 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_180 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_207 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_210 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_215 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_220 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_230 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_235 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_247 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_275 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_280 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_287 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_295 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_300 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_320 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_327 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_330 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_347 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_360 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_367 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_370 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_375 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_380 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_387 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_390 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_395 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_400 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_407 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_410 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_420 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_427 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_430 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_440 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_447 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_450 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_455 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_460 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_467 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_470 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_480 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_487 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_490 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_495 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_500 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_507 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_510 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_515 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_38 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_520 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_527 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_530 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_535 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_540 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_547 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_550 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_555 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_560 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_567 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_570 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_575 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_44 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_580 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_587 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_590 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_595 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_600 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_607 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_610 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_615 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_620 = rand_strided((196, 192), (192, 1), device='cpu', dtype=torch.float32)
    permute_627 = rand_strided((384, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_159, primals_165, primals_171, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_231, primals_237, primals_243, primals_249, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, primals_295, mul, view_1, getitem_2, getitem_3, view_3, mul_4, view_5, getitem_6, getitem_7, view_7, mul_8, view_9, getitem_10, getitem_11, view_11, mul_12, view_13, getitem_14, getitem_15, view_15, mul_16, view_17, getitem_18, getitem_19, view_19, mul_20, view_21, getitem_22, getitem_23, view_23, mul_24, view_25, getitem_26, getitem_27, view_27, mul_28, view_29, getitem_30, getitem_31, view_31, mul_32, view_33, getitem_34, getitem_35, view_35, mul_36, view_37, getitem_38, getitem_39, view_39, mul_40, view_41, getitem_42, getitem_43, view_43, mul_44, view_45, getitem_46, getitem_47, view_47, mul_48, view_49, getitem_50, getitem_51, view_51, mul_52, view_53, getitem_54, getitem_55, view_55, mul_56, view_57, getitem_58, getitem_59, view_59, mul_60, view_61, getitem_62, getitem_63, view_63, mul_64, view_65, getitem_66, getitem_67, view_67, mul_68, view_69, getitem_70, getitem_71, view_71, mul_72, view_73, getitem_74, getitem_75, view_75, mul_76, view_77, getitem_78, getitem_79, view_79, mul_80, view_81, getitem_82, getitem_83, view_83, mul_84, view_85, getitem_86, getitem_87, view_87, mul_88, view_89, getitem_90, getitem_91, view_91, mul_92, view_93, getitem_94, getitem_95, view_95, mul_96, view_97, getitem_98, getitem_99, view_99, mul_100, view_101, getitem_102, getitem_103, view_103, mul_104, view_105, getitem_106, getitem_107, view_107, mul_108, view_109, getitem_110, getitem_111, view_111, mul_112, view_113, getitem_114, getitem_115, view_115, mul_116, view_117, getitem_118, getitem_119, view_119, mul_120, view_121, getitem_122, getitem_123, view_123, mul_124, view_125, getitem_126, getitem_127, view_127, mul_128, view_129, getitem_130, getitem_131, view_131, mul_132, view_133, getitem_134, getitem_135, view_135, mul_136, view_137, getitem_138, getitem_139, view_139, mul_140, view_141, getitem_142, getitem_143, view_143, mul_144, view_145, getitem_146, getitem_147, view_147, mul_148, view_149, getitem_150, getitem_151, view_151, mul_152, view_153, getitem_154, getitem_155, view_155, mul_156, view_157, getitem_158, getitem_159, view_159, mul_160, view_161, getitem_162, getitem_163, view_163, mul_164, view_165, getitem_166, getitem_167, view_167, mul_168, view_169, getitem_170, getitem_171, view_171, mul_172, view_173, getitem_174, getitem_175, view_175, mul_176, view_177, getitem_178, getitem_179, view_179, mul_180, view_181, getitem_182, getitem_183, view_183, mul_184, view_185, getitem_186, getitem_187, view_187, mul_188, view_189, getitem_190, getitem_191, view_191, mul_192, clone_169, permute_146, div_1, permute_150, permute_155, div_2, permute_160, permute_167, div_3, permute_170, permute_175, div_4, permute_180, permute_187, div_5, permute_190, permute_195, div_6, permute_200, permute_207, div_7, permute_210, permute_215, div_8, permute_220, permute_227, div_9, permute_230, permute_235, div_10, permute_240, permute_247, div_11, permute_250, permute_255, div_12, permute_260, permute_267, div_13, permute_270, permute_275, div_14, permute_280, permute_287, div_15, permute_290, permute_295, div_16, permute_300, permute_307, div_17, permute_310, permute_315, div_18, permute_320, permute_327, div_19, permute_330, permute_335, div_20, permute_340, permute_347, div_21, permute_350, permute_355, div_22, permute_360, permute_367, div_23, permute_370, permute_375, div_24, permute_380, permute_387, div_25, permute_390, permute_395, div_26, permute_400, permute_407, div_27, permute_410, permute_415, div_28, permute_420, permute_427, div_29, permute_430, permute_435, div_30, permute_440, permute_447, div_31, permute_450, permute_455, div_32, permute_460, permute_467, div_33, permute_470, permute_475, div_34, permute_480, permute_487, div_35, permute_490, permute_495, div_36, permute_500, permute_507, div_37, permute_510, permute_515, div_38, permute_520, permute_527, div_39, permute_530, permute_535, div_40, permute_540, permute_547, div_41, permute_550, permute_555, div_42, permute_560, permute_567, div_43, permute_570, permute_575, div_44, permute_580, permute_587, div_45, permute_590, permute_595, div_46, permute_600, permute_607, div_47, permute_610, permute_615, div_48, permute_620, permute_627, div_49, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmixer_24_224', benchmark_compiled_module)
