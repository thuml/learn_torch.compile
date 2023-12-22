
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50176L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (50176L*x0)));
                        auto tmp14 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = static_cast<float>(196.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp7 = static_cast<float>(256.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 - tmp11;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp12 - tmp16;
                        auto tmp18 = at::vec::Vectorized<float>(tmp0);
                        auto tmp19 = tmp18 * tmp17;
                        tmp19.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (50176L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x2) + (50176L*x1)));
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


cpp_fused_clone_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_2 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_3 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_5 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_6 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_8 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_9 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_11 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_12 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_14 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_15 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_17 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_18 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_20 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_21 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_23 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_24 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_26 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_27 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_29 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_30 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_32 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_33 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_35 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_36 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_38 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_39 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_41 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_42 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_44 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_45 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_47 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_48 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_50 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_51 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_53 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_54 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_56 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_57 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_59 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_60 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_62 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_63 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_65 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_66 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_68 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_69 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_71 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_72 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_74 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_75 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_77 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_78 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_80 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_81 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_83 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_84 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_86 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_87 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0) + (150528L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0) + (301056L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (1536L*x1_inner) + (301056L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (150528L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (150528L*x0))] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_89 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)), static_cast<long>(768L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (150528L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2)];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (150528L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (768L*x2_inner) + (150528L*x1)));
                                auto tmp3 = tmp1 * tmp2;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp1;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (150528L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (150528L*x0))];
                            auto tmp7 = in_ptr4[static_cast<long>(x1 + (196L*x2) + (150528L*x0))];
                            auto tmp8 = in_ptr5[static_cast<long>(x1)];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
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
                            auto tmp16 = in_ptr6[static_cast<long>(x1 + (196L*x0))];
                            auto tmp17 = in_ptr0[static_cast<long>((-150528L) + x1 + (196L*x2) + (150528L*x0))];
                            auto tmp18 = in_ptr1[static_cast<long>((-768L) + x2)];
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = static_cast<float>(768.0);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                            auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                            auto tmp24 = in_ptr2[static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0))];
                            auto tmp25 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp23)(tmp23 - tmp26);
                            auto tmp28 = decltype(tmp16)(tmp16 * tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp30 = tmp4 ? tmp11 : tmp29;
                        out_ptr4[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))] = tmp30;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_layer_norm_backward_sum_90 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_7, primals_10, primals_13, primals_17, primals_20, primals_23, primals_27, primals_30, primals_33, primals_37, primals_40, primals_43, primals_47, primals_50, primals_53, primals_57, primals_60, primals_63, primals_67, primals_70, primals_73, primals_77, primals_80, primals_83, primals_87, primals_90, primals_93, primals_97, primals_100, primals_103, primals_107, primals_110, primals_113, primals_117, primals_120, primals_123, primals_127, primals_130, primals_133, primals_137, primals_140, primals_143, primals_147, primals_150, primals_153, primals_157, primals_160, primals_163, primals_167, primals_170, primals_173, primals_177, primals_180, primals_183, primals_187, primals_190, primals_193, primals_197, primals_200, primals_203, primals_207, primals_210, primals_213, primals_217, primals_220, primals_223, primals_227, primals_230, primals_233, primals_237, primals_240, primals_243, primals_247, primals_250, primals_253, primals_257, primals_260, primals_263, primals_267, primals_270, primals_273, primals_277, primals_280, primals_283, primals_287, primals_290, primals_293, primals_297, primals_300, primals_303, primals_307, mul, view_1, addmm, getitem_2, mul_5, view_3, mm, view_5, mul_8, view_7, addmm_2, getitem_8, mul_13, view_9, mm_1, view_11, mul_16, view_13, addmm_4, getitem_14, mul_21, view_15, mm_2, view_17, mul_24, view_19, addmm_6, getitem_20, mul_29, view_21, mm_3, view_23, mul_32, view_25, addmm_8, getitem_26, mul_37, view_27, mm_4, view_29, mul_40, view_31, addmm_10, getitem_32, mul_45, view_33, mm_5, view_35, mul_48, view_37, addmm_12, getitem_38, mul_53, view_39, mm_6, view_41, mul_56, view_43, addmm_14, getitem_44, mul_61, view_45, mm_7, view_47, mul_64, view_49, addmm_16, getitem_50, mul_69, view_51, mm_8, view_53, mul_72, view_55, addmm_18, getitem_56, mul_77, view_57, mm_9, view_59, mul_80, view_61, addmm_20, getitem_62, mul_85, view_63, mm_10, view_65, mul_88, view_67, addmm_22, getitem_68, mul_93, view_69, mm_11, view_71, mul_96, view_73, addmm_24, getitem_74, mul_101, view_75, mm_12, view_77, mul_104, view_79, addmm_26, getitem_80, mul_109, view_81, mm_13, view_83, mul_112, view_85, addmm_28, getitem_86, mul_117, view_87, mm_14, view_89, mul_120, view_91, addmm_30, getitem_92, mul_125, view_93, mm_15, view_95, mul_128, view_97, addmm_32, getitem_98, mul_133, view_99, mm_16, view_101, mul_136, view_103, addmm_34, getitem_104, mul_141, view_105, mm_17, view_107, mul_144, view_109, addmm_36, getitem_110, mul_149, view_111, mm_18, view_113, mul_152, view_115, addmm_38, getitem_116, mul_157, view_117, mm_19, view_119, mul_160, view_121, addmm_40, getitem_122, mul_165, view_123, mm_20, view_125, mul_168, view_127, addmm_42, getitem_128, mul_173, view_129, mm_21, view_131, mul_176, view_133, addmm_44, getitem_134, mul_181, view_135, mm_22, view_137, mul_184, view_139, addmm_46, getitem_140, mul_189, view_141, mm_23, view_143, mul_192, view_145, addmm_48, getitem_146, mul_197, view_147, mm_24, view_149, mul_200, view_151, addmm_50, getitem_152, mul_205, view_153, mm_25, view_155, mul_208, view_157, addmm_52, getitem_158, mul_213, view_159, mm_26, view_161, mul_216, view_163, addmm_54, getitem_164, mul_221, view_165, mm_27, view_167, mul_224, view_169, addmm_56, getitem_170, mul_229, view_171, mm_28, view_173, mul_232, view_175, addmm_58, getitem_176, mul_237, view_177, mm_29, view_179, mul_240, clone_151, permute_152, div_1, permute_156, permute_163, div_2, permute_166, div_3, permute_170, permute_177, div_4, permute_180, div_5, permute_184, permute_191, div_6, permute_194, div_7, permute_198, permute_205, div_8, permute_208, div_9, permute_212, permute_219, div_10, permute_222, div_11, permute_226, permute_233, div_12, permute_236, div_13, permute_240, permute_247, div_14, permute_250, div_15, permute_254, permute_261, div_16, permute_264, div_17, permute_268, permute_275, div_18, permute_278, div_19, permute_282, permute_289, div_20, permute_292, div_21, permute_296, permute_303, div_22, permute_306, div_23, permute_310, permute_317, div_24, permute_320, div_25, permute_324, permute_331, div_26, permute_334, div_27, permute_338, permute_345, div_28, permute_348, div_29, permute_352, permute_359, div_30, permute_362, div_31, permute_366, permute_373, div_32, permute_376, div_33, permute_380, permute_387, div_34, permute_390, div_35, permute_394, permute_401, div_36, permute_404, div_37, permute_408, permute_415, div_38, permute_418, div_39, permute_422, permute_429, div_40, permute_432, div_41, permute_436, permute_443, div_42, permute_446, div_43, permute_450, permute_457, div_44, permute_460, div_45, permute_464, permute_471, div_46, permute_474, div_47, permute_478, permute_485, div_48, permute_488, div_49, permute_492, permute_499, div_50, permute_502, div_51, permute_506, permute_513, div_52, permute_516, div_53, permute_520, permute_527, div_54, permute_530, div_55, permute_534, permute_541, div_56, permute_544, div_57, permute_548, permute_555, div_58, permute_558, div_59, permute_562, permute_569, div_60, permute_572, div_61, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (256, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_10, (196, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_20, (196, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_30, (196, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_40, (196, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_50, (196, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_60, (196, ), (1, ))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_70, (196, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_80, (196, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_90, (196, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_100, (196, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_110, (196, ), (1, ))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_120, (196, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_130, (196, ), (1, ))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_140, (196, ), (1, ))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_150, (196, ), (1, ))
    assert_size_stride(primals_153, (256, ), (1, ))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_160, (196, ), (1, ))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_170, (196, ), (1, ))
    assert_size_stride(primals_173, (256, ), (1, ))
    assert_size_stride(primals_177, (768, ), (1, ))
    assert_size_stride(primals_180, (196, ), (1, ))
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_190, (196, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_200, (196, ), (1, ))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_210, (196, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_220, (196, ), (1, ))
    assert_size_stride(primals_223, (256, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_230, (196, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_237, (768, ), (1, ))
    assert_size_stride(primals_240, (196, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_250, (196, ), (1, ))
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_257, (768, ), (1, ))
    assert_size_stride(primals_260, (196, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_270, (196, ), (1, ))
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_277, (768, ), (1, ))
    assert_size_stride(primals_280, (196, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_290, (196, ), (1, ))
    assert_size_stride(primals_293, (256, ), (1, ))
    assert_size_stride(primals_297, (768, ), (1, ))
    assert_size_stride(primals_300, (196, ), (1, ))
    assert_size_stride(primals_303, (256, ), (1, ))
    assert_size_stride(primals_307, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_1, (1568, 256), (256, 1))
    assert_size_stride(addmm, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_2, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_5, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_3, (6144, 196), (196, 1))
    assert_size_stride(mm, (6144, 196), (196, 1))
    assert_size_stride(view_5, (1568, 768), (768, 1))
    assert_size_stride(mul_8, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_7, (1568, 256), (256, 1))
    assert_size_stride(addmm_2, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_8, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_13, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_9, (6144, 196), (196, 1))
    assert_size_stride(mm_1, (6144, 196), (196, 1))
    assert_size_stride(view_11, (1568, 768), (768, 1))
    assert_size_stride(mul_16, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_13, (1568, 256), (256, 1))
    assert_size_stride(addmm_4, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_14, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_21, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_15, (6144, 196), (196, 1))
    assert_size_stride(mm_2, (6144, 196), (196, 1))
    assert_size_stride(view_17, (1568, 768), (768, 1))
    assert_size_stride(mul_24, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_19, (1568, 256), (256, 1))
    assert_size_stride(addmm_6, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_20, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_29, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_21, (6144, 196), (196, 1))
    assert_size_stride(mm_3, (6144, 196), (196, 1))
    assert_size_stride(view_23, (1568, 768), (768, 1))
    assert_size_stride(mul_32, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_25, (1568, 256), (256, 1))
    assert_size_stride(addmm_8, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_26, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_37, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_27, (6144, 196), (196, 1))
    assert_size_stride(mm_4, (6144, 196), (196, 1))
    assert_size_stride(view_29, (1568, 768), (768, 1))
    assert_size_stride(mul_40, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_31, (1568, 256), (256, 1))
    assert_size_stride(addmm_10, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_32, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_45, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_33, (6144, 196), (196, 1))
    assert_size_stride(mm_5, (6144, 196), (196, 1))
    assert_size_stride(view_35, (1568, 768), (768, 1))
    assert_size_stride(mul_48, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_37, (1568, 256), (256, 1))
    assert_size_stride(addmm_12, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_38, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_53, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_39, (6144, 196), (196, 1))
    assert_size_stride(mm_6, (6144, 196), (196, 1))
    assert_size_stride(view_41, (1568, 768), (768, 1))
    assert_size_stride(mul_56, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_43, (1568, 256), (256, 1))
    assert_size_stride(addmm_14, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_44, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_61, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_45, (6144, 196), (196, 1))
    assert_size_stride(mm_7, (6144, 196), (196, 1))
    assert_size_stride(view_47, (1568, 768), (768, 1))
    assert_size_stride(mul_64, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_49, (1568, 256), (256, 1))
    assert_size_stride(addmm_16, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_50, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_69, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_51, (6144, 196), (196, 1))
    assert_size_stride(mm_8, (6144, 196), (196, 1))
    assert_size_stride(view_53, (1568, 768), (768, 1))
    assert_size_stride(mul_72, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_55, (1568, 256), (256, 1))
    assert_size_stride(addmm_18, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_56, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_77, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_57, (6144, 196), (196, 1))
    assert_size_stride(mm_9, (6144, 196), (196, 1))
    assert_size_stride(view_59, (1568, 768), (768, 1))
    assert_size_stride(mul_80, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_61, (1568, 256), (256, 1))
    assert_size_stride(addmm_20, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_62, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_85, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_63, (6144, 196), (196, 1))
    assert_size_stride(mm_10, (6144, 196), (196, 1))
    assert_size_stride(view_65, (1568, 768), (768, 1))
    assert_size_stride(mul_88, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_67, (1568, 256), (256, 1))
    assert_size_stride(addmm_22, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_68, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_93, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_69, (6144, 196), (196, 1))
    assert_size_stride(mm_11, (6144, 196), (196, 1))
    assert_size_stride(view_71, (1568, 768), (768, 1))
    assert_size_stride(mul_96, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_73, (1568, 256), (256, 1))
    assert_size_stride(addmm_24, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_74, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_101, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_75, (6144, 196), (196, 1))
    assert_size_stride(mm_12, (6144, 196), (196, 1))
    assert_size_stride(view_77, (1568, 768), (768, 1))
    assert_size_stride(mul_104, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_79, (1568, 256), (256, 1))
    assert_size_stride(addmm_26, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_80, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_109, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_81, (6144, 196), (196, 1))
    assert_size_stride(mm_13, (6144, 196), (196, 1))
    assert_size_stride(view_83, (1568, 768), (768, 1))
    assert_size_stride(mul_112, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_85, (1568, 256), (256, 1))
    assert_size_stride(addmm_28, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_86, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_117, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_87, (6144, 196), (196, 1))
    assert_size_stride(mm_14, (6144, 196), (196, 1))
    assert_size_stride(view_89, (1568, 768), (768, 1))
    assert_size_stride(mul_120, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_91, (1568, 256), (256, 1))
    assert_size_stride(addmm_30, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_92, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_125, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_93, (6144, 196), (196, 1))
    assert_size_stride(mm_15, (6144, 196), (196, 1))
    assert_size_stride(view_95, (1568, 768), (768, 1))
    assert_size_stride(mul_128, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_97, (1568, 256), (256, 1))
    assert_size_stride(addmm_32, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_98, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_133, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_99, (6144, 196), (196, 1))
    assert_size_stride(mm_16, (6144, 196), (196, 1))
    assert_size_stride(view_101, (1568, 768), (768, 1))
    assert_size_stride(mul_136, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_103, (1568, 256), (256, 1))
    assert_size_stride(addmm_34, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_104, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_141, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_105, (6144, 196), (196, 1))
    assert_size_stride(mm_17, (6144, 196), (196, 1))
    assert_size_stride(view_107, (1568, 768), (768, 1))
    assert_size_stride(mul_144, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_109, (1568, 256), (256, 1))
    assert_size_stride(addmm_36, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_110, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_149, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_111, (6144, 196), (196, 1))
    assert_size_stride(mm_18, (6144, 196), (196, 1))
    assert_size_stride(view_113, (1568, 768), (768, 1))
    assert_size_stride(mul_152, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_115, (1568, 256), (256, 1))
    assert_size_stride(addmm_38, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_116, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_157, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_117, (6144, 196), (196, 1))
    assert_size_stride(mm_19, (6144, 196), (196, 1))
    assert_size_stride(view_119, (1568, 768), (768, 1))
    assert_size_stride(mul_160, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_121, (1568, 256), (256, 1))
    assert_size_stride(addmm_40, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_122, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_165, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_123, (6144, 196), (196, 1))
    assert_size_stride(mm_20, (6144, 196), (196, 1))
    assert_size_stride(view_125, (1568, 768), (768, 1))
    assert_size_stride(mul_168, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_127, (1568, 256), (256, 1))
    assert_size_stride(addmm_42, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_128, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_173, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_129, (6144, 196), (196, 1))
    assert_size_stride(mm_21, (6144, 196), (196, 1))
    assert_size_stride(view_131, (1568, 768), (768, 1))
    assert_size_stride(mul_176, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_133, (1568, 256), (256, 1))
    assert_size_stride(addmm_44, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_134, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_181, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_135, (6144, 196), (196, 1))
    assert_size_stride(mm_22, (6144, 196), (196, 1))
    assert_size_stride(view_137, (1568, 768), (768, 1))
    assert_size_stride(mul_184, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_139, (1568, 256), (256, 1))
    assert_size_stride(addmm_46, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_140, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_189, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_141, (6144, 196), (196, 1))
    assert_size_stride(mm_23, (6144, 196), (196, 1))
    assert_size_stride(view_143, (1568, 768), (768, 1))
    assert_size_stride(mul_192, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_145, (1568, 256), (256, 1))
    assert_size_stride(addmm_48, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_146, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_197, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_147, (6144, 196), (196, 1))
    assert_size_stride(mm_24, (6144, 196), (196, 1))
    assert_size_stride(view_149, (1568, 768), (768, 1))
    assert_size_stride(mul_200, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_151, (1568, 256), (256, 1))
    assert_size_stride(addmm_50, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_152, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_205, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_153, (6144, 196), (196, 1))
    assert_size_stride(mm_25, (6144, 196), (196, 1))
    assert_size_stride(view_155, (1568, 768), (768, 1))
    assert_size_stride(mul_208, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_157, (1568, 256), (256, 1))
    assert_size_stride(addmm_52, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_158, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_213, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_159, (6144, 196), (196, 1))
    assert_size_stride(mm_26, (6144, 196), (196, 1))
    assert_size_stride(view_161, (1568, 768), (768, 1))
    assert_size_stride(mul_216, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_163, (1568, 256), (256, 1))
    assert_size_stride(addmm_54, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_164, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_221, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_165, (6144, 196), (196, 1))
    assert_size_stride(mm_27, (6144, 196), (196, 1))
    assert_size_stride(view_167, (1568, 768), (768, 1))
    assert_size_stride(mul_224, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_169, (1568, 256), (256, 1))
    assert_size_stride(addmm_56, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_170, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_229, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_171, (6144, 196), (196, 1))
    assert_size_stride(mm_28, (6144, 196), (196, 1))
    assert_size_stride(view_173, (1568, 768), (768, 1))
    assert_size_stride(mul_232, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_175, (1568, 256), (256, 1))
    assert_size_stride(addmm_58, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_176, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_237, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_177, (6144, 196), (196, 1))
    assert_size_stride(mm_29, (6144, 196), (196, 1))
    assert_size_stride(view_179, (1568, 768), (768, 1))
    assert_size_stride(mul_240, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(clone_151, (8, 256), (256, 1))
    assert_size_stride(permute_152, (1000, 256), (256, 1))
    assert_size_stride(div_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_156, (256, 768), (768, 1))
    assert_size_stride(permute_163, (196, 196), (196, 1))
    assert_size_stride(div_2, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_166, (1536, 256), (256, 1))
    assert_size_stride(div_3, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_170, (256, 768), (768, 1))
    assert_size_stride(permute_177, (196, 196), (196, 1))
    assert_size_stride(div_4, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_180, (1536, 256), (256, 1))
    assert_size_stride(div_5, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_184, (256, 768), (768, 1))
    assert_size_stride(permute_191, (196, 196), (196, 1))
    assert_size_stride(div_6, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_194, (1536, 256), (256, 1))
    assert_size_stride(div_7, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_198, (256, 768), (768, 1))
    assert_size_stride(permute_205, (196, 196), (196, 1))
    assert_size_stride(div_8, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_208, (1536, 256), (256, 1))
    assert_size_stride(div_9, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_212, (256, 768), (768, 1))
    assert_size_stride(permute_219, (196, 196), (196, 1))
    assert_size_stride(div_10, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_222, (1536, 256), (256, 1))
    assert_size_stride(div_11, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_226, (256, 768), (768, 1))
    assert_size_stride(permute_233, (196, 196), (196, 1))
    assert_size_stride(div_12, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_236, (1536, 256), (256, 1))
    assert_size_stride(div_13, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (256, 768), (768, 1))
    assert_size_stride(permute_247, (196, 196), (196, 1))
    assert_size_stride(div_14, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_250, (1536, 256), (256, 1))
    assert_size_stride(div_15, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_254, (256, 768), (768, 1))
    assert_size_stride(permute_261, (196, 196), (196, 1))
    assert_size_stride(div_16, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_264, (1536, 256), (256, 1))
    assert_size_stride(div_17, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_268, (256, 768), (768, 1))
    assert_size_stride(permute_275, (196, 196), (196, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_278, (1536, 256), (256, 1))
    assert_size_stride(div_19, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_282, (256, 768), (768, 1))
    assert_size_stride(permute_289, (196, 196), (196, 1))
    assert_size_stride(div_20, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_292, (1536, 256), (256, 1))
    assert_size_stride(div_21, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_296, (256, 768), (768, 1))
    assert_size_stride(permute_303, (196, 196), (196, 1))
    assert_size_stride(div_22, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_306, (1536, 256), (256, 1))
    assert_size_stride(div_23, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_310, (256, 768), (768, 1))
    assert_size_stride(permute_317, (196, 196), (196, 1))
    assert_size_stride(div_24, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_320, (1536, 256), (256, 1))
    assert_size_stride(div_25, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_324, (256, 768), (768, 1))
    assert_size_stride(permute_331, (196, 196), (196, 1))
    assert_size_stride(div_26, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_334, (1536, 256), (256, 1))
    assert_size_stride(div_27, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_338, (256, 768), (768, 1))
    assert_size_stride(permute_345, (196, 196), (196, 1))
    assert_size_stride(div_28, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_348, (1536, 256), (256, 1))
    assert_size_stride(div_29, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_352, (256, 768), (768, 1))
    assert_size_stride(permute_359, (196, 196), (196, 1))
    assert_size_stride(div_30, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_362, (1536, 256), (256, 1))
    assert_size_stride(div_31, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_366, (256, 768), (768, 1))
    assert_size_stride(permute_373, (196, 196), (196, 1))
    assert_size_stride(div_32, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_376, (1536, 256), (256, 1))
    assert_size_stride(div_33, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_380, (256, 768), (768, 1))
    assert_size_stride(permute_387, (196, 196), (196, 1))
    assert_size_stride(div_34, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_390, (1536, 256), (256, 1))
    assert_size_stride(div_35, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_394, (256, 768), (768, 1))
    assert_size_stride(permute_401, (196, 196), (196, 1))
    assert_size_stride(div_36, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_404, (1536, 256), (256, 1))
    assert_size_stride(div_37, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_408, (256, 768), (768, 1))
    assert_size_stride(permute_415, (196, 196), (196, 1))
    assert_size_stride(div_38, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_418, (1536, 256), (256, 1))
    assert_size_stride(div_39, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_422, (256, 768), (768, 1))
    assert_size_stride(permute_429, (196, 196), (196, 1))
    assert_size_stride(div_40, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_432, (1536, 256), (256, 1))
    assert_size_stride(div_41, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_436, (256, 768), (768, 1))
    assert_size_stride(permute_443, (196, 196), (196, 1))
    assert_size_stride(div_42, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_446, (1536, 256), (256, 1))
    assert_size_stride(div_43, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_450, (256, 768), (768, 1))
    assert_size_stride(permute_457, (196, 196), (196, 1))
    assert_size_stride(div_44, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_460, (1536, 256), (256, 1))
    assert_size_stride(div_45, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_464, (256, 768), (768, 1))
    assert_size_stride(permute_471, (196, 196), (196, 1))
    assert_size_stride(div_46, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_474, (1536, 256), (256, 1))
    assert_size_stride(div_47, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_478, (256, 768), (768, 1))
    assert_size_stride(permute_485, (196, 196), (196, 1))
    assert_size_stride(div_48, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_488, (1536, 256), (256, 1))
    assert_size_stride(div_49, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_492, (256, 768), (768, 1))
    assert_size_stride(permute_499, (196, 196), (196, 1))
    assert_size_stride(div_50, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_502, (1536, 256), (256, 1))
    assert_size_stride(div_51, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_506, (256, 768), (768, 1))
    assert_size_stride(permute_513, (196, 196), (196, 1))
    assert_size_stride(div_52, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_516, (1536, 256), (256, 1))
    assert_size_stride(div_53, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_520, (256, 768), (768, 1))
    assert_size_stride(permute_527, (196, 196), (196, 1))
    assert_size_stride(div_54, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_530, (1536, 256), (256, 1))
    assert_size_stride(div_55, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_534, (256, 768), (768, 1))
    assert_size_stride(permute_541, (196, 196), (196, 1))
    assert_size_stride(div_56, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_544, (1536, 256), (256, 1))
    assert_size_stride(div_57, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_548, (256, 768), (768, 1))
    assert_size_stride(permute_555, (196, 196), (196, 1))
    assert_size_stride(div_58, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_558, (1536, 256), (256, 1))
    assert_size_stride(div_59, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_562, (256, 768), (768, 1))
    assert_size_stride(permute_569, (196, 196), (196, 1))
    assert_size_stride(div_60, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_572, (1536, 256), (256, 1))
    assert_size_stride(div_61, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_152, out=buf0)
    del permute_152
    buf1 = empty((1000, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_151, out=buf1)
    del clone_151
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    buf6 = empty((256, ), device='cpu', dtype=torch.float32)
    buf7 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_div_native_layer_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(mul_240.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf0
    del div_1
    del mul_240
    del primals_303
    del tangents_1
    buf8 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1568, 256), (256, 1), 0), permute_156, out=buf8)
    del permute_156
    buf9 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (256, 1568), (1, 256), 0), view_179, out=buf9)
    del view_179
    buf10 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf11 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf12 = empty((8, 768, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_sum_1(c_void_p(buf5.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(getitem_176.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del getitem_176
    buf13 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (196, 6144), (1, 196), 0), view_177, out=buf13)
    del view_177
    buf14 = empty((6144, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (6144, 196), (196, 1), 0), permute_163, out=buf14)
    del permute_163
    buf15 = buf4; del buf4  # reuse
    buf16 = buf3; del buf3  # reuse
    buf17 = empty((768, ), device='cpu', dtype=torch.float32)
    buf18 = empty((768, ), device='cpu', dtype=torch.float32)
    buf19 = empty((8, 196, 1536), device='cpu', dtype=torch.float32)
    buf20 = buf19; del buf19  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_2(c_void_p(buf20.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(mul_237.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(mm_29.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del addmm_58
    del div_2
    del mm_29
    del mul_237
    del primals_297
    del primals_300
    buf21 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (1568, 1536), (1536, 1), 0), permute_166, out=buf21)
    del permute_166
    buf22 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (1536, 1568), (1, 1536), 0), view_175, out=buf22)
    del view_175
    buf23 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf24 = buf16; del buf16  # reuse
    buf25 = buf15; del buf15  # reuse
    buf26 = empty((256, ), device='cpu', dtype=torch.float32)
    buf27 = empty((256, ), device='cpu', dtype=torch.float32)
    buf28 = reinterpret_tensor(buf21, (8, 196, 256), (50176, 256, 1), 0); del buf21  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_3(c_void_p(buf28.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(mul_232.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del div_3
    del mul_232
    del primals_293
    buf29 = buf8; del buf8  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (1568, 256), (256, 1), 0), permute_170, out=buf29)
    del permute_170
    buf30 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (256, 1568), (1, 256), 0), view_173, out=buf30)
    del view_173
    buf31 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf32 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf33 = reinterpret_tensor(buf14, (8, 768, 196), (150528, 196, 1), 0); del buf14  # reuse
    cpp_fused_clone_sum_4(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(getitem_170.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del getitem_170
    buf34 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf33, (196, 6144), (1, 196), 0), view_171, out=buf34)
    del view_171
    buf35 = reinterpret_tensor(buf12, (6144, 196), (196, 1), 0); del buf12  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf33, (6144, 196), (196, 1), 0), permute_177, out=buf35)
    del permute_177
    buf36 = buf25; del buf25  # reuse
    buf37 = buf24; del buf24  # reuse
    buf38 = empty((768, ), device='cpu', dtype=torch.float32)
    buf39 = empty((768, ), device='cpu', dtype=torch.float32)
    buf40 = buf20; del buf20  # reuse
    buf41 = buf40; del buf40  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_5(c_void_p(buf41.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(mul_229.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(mm_28.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(addmm_56.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    del addmm_56
    del div_4
    del mm_28
    del mul_229
    del primals_287
    del primals_290
    buf42 = reinterpret_tensor(buf5, (1568, 256), (256, 1), 0); del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf41, (1568, 1536), (1536, 1), 0), permute_180, out=buf42)
    del permute_180
    buf43 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf41, (1536, 1568), (1, 1536), 0), view_169, out=buf43)
    del view_169
    buf44 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf45 = buf37; del buf37  # reuse
    buf46 = buf36; del buf36  # reuse
    buf47 = empty((256, ), device='cpu', dtype=torch.float32)
    buf48 = empty((256, ), device='cpu', dtype=torch.float32)
    buf49 = buf28; del buf28  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_6(c_void_p(buf49.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(mul_224.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del div_5
    del mul_224
    del primals_283
    buf50 = reinterpret_tensor(buf35, (1568, 768), (768, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (1568, 256), (256, 1), 0), permute_184, out=buf50)
    del permute_184
    buf51 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (256, 1568), (1, 256), 0), view_167, out=buf51)
    del view_167
    buf52 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf53 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf54 = reinterpret_tensor(buf29, (8, 768, 196), (150528, 196, 1), 0); del buf29  # reuse
    cpp_fused_clone_sum_7(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(getitem_164.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del getitem_164
    buf55 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf54, (196, 6144), (1, 196), 0), view_165, out=buf55)
    del view_165
    buf56 = reinterpret_tensor(buf33, (6144, 196), (196, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf54, (6144, 196), (196, 1), 0), permute_191, out=buf56)
    del permute_191
    buf57 = buf46; del buf46  # reuse
    buf58 = buf45; del buf45  # reuse
    buf59 = empty((768, ), device='cpu', dtype=torch.float32)
    buf60 = empty((768, ), device='cpu', dtype=torch.float32)
    buf61 = buf41; del buf41  # reuse
    buf62 = buf61; del buf61  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_8(c_void_p(buf62.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(mul_221.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(mm_27.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(addmm_54.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del addmm_54
    del div_6
    del mm_27
    del mul_221
    del primals_277
    del primals_280
    buf63 = buf42; del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (1568, 1536), (1536, 1), 0), permute_194, out=buf63)
    del permute_194
    buf64 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (1536, 1568), (1, 1536), 0), view_163, out=buf64)
    del view_163
    buf65 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf66 = buf58; del buf58  # reuse
    buf67 = buf57; del buf57  # reuse
    buf68 = empty((256, ), device='cpu', dtype=torch.float32)
    buf69 = empty((256, ), device='cpu', dtype=torch.float32)
    buf70 = buf49; del buf49  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_9(c_void_p(buf70.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(mul_216.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del div_7
    del mul_216
    del primals_273
    buf71 = reinterpret_tensor(buf56, (1568, 768), (768, 1), 0); del buf56  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (1568, 256), (256, 1), 0), permute_198, out=buf71)
    del permute_198
    buf72 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (256, 1568), (1, 256), 0), view_161, out=buf72)
    del view_161
    buf73 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf74 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf75 = reinterpret_tensor(buf50, (8, 768, 196), (150528, 196, 1), 0); del buf50  # reuse
    cpp_fused_clone_sum_10(c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(getitem_158.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del getitem_158
    buf76 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (196, 6144), (1, 196), 0), view_159, out=buf76)
    del view_159
    buf77 = reinterpret_tensor(buf54, (6144, 196), (196, 1), 0); del buf54  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (6144, 196), (196, 1), 0), permute_205, out=buf77)
    del permute_205
    buf78 = buf67; del buf67  # reuse
    buf79 = buf66; del buf66  # reuse
    buf80 = empty((768, ), device='cpu', dtype=torch.float32)
    buf81 = empty((768, ), device='cpu', dtype=torch.float32)
    buf82 = buf62; del buf62  # reuse
    buf83 = buf82; del buf82  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_11(c_void_p(buf83.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(mul_213.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(mm_26.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(div_8.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del addmm_52
    del div_8
    del mm_26
    del mul_213
    del primals_267
    del primals_270
    buf84 = buf63; del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (1568, 1536), (1536, 1), 0), permute_208, out=buf84)
    del permute_208
    buf85 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (1536, 1568), (1, 1536), 0), view_157, out=buf85)
    del view_157
    buf86 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf87 = buf79; del buf79  # reuse
    buf88 = buf78; del buf78  # reuse
    buf89 = empty((256, ), device='cpu', dtype=torch.float32)
    buf90 = empty((256, ), device='cpu', dtype=torch.float32)
    buf91 = buf70; del buf70  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_12(c_void_p(buf91.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(mul_208.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del div_9
    del mul_208
    del primals_263
    buf92 = reinterpret_tensor(buf77, (1568, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (1568, 256), (256, 1), 0), permute_212, out=buf92)
    del permute_212
    buf93 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (256, 1568), (1, 256), 0), view_155, out=buf93)
    del view_155
    buf94 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf95 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf96 = reinterpret_tensor(buf71, (8, 768, 196), (150528, 196, 1), 0); del buf71  # reuse
    cpp_fused_clone_sum_13(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(getitem_152.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del getitem_152
    buf97 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (196, 6144), (1, 196), 0), view_153, out=buf97)
    del view_153
    buf98 = reinterpret_tensor(buf75, (6144, 196), (196, 1), 0); del buf75  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (6144, 196), (196, 1), 0), permute_219, out=buf98)
    del permute_219
    buf99 = buf88; del buf88  # reuse
    buf100 = buf87; del buf87  # reuse
    buf101 = empty((768, ), device='cpu', dtype=torch.float32)
    buf102 = empty((768, ), device='cpu', dtype=torch.float32)
    buf103 = buf83; del buf83  # reuse
    buf104 = buf103; del buf103  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_14(c_void_p(buf104.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(mul_205.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(mm_25.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(addmm_50.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del addmm_50
    del div_10
    del mm_25
    del mul_205
    del primals_257
    del primals_260
    buf105 = buf84; del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (1568, 1536), (1536, 1), 0), permute_222, out=buf105)
    del permute_222
    buf106 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (1536, 1568), (1, 1536), 0), view_151, out=buf106)
    del view_151
    buf107 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf108 = buf99; del buf99  # reuse
    buf109 = buf100; del buf100  # reuse
    buf110 = empty((256, ), device='cpu', dtype=torch.float32)
    buf111 = empty((256, ), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf105, (8, 196, 256), (50176, 256, 1), 0); del buf105  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_15(c_void_p(buf112.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(mul_200.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del div_11
    del mul_200
    del primals_253
    buf113 = reinterpret_tensor(buf98, (1568, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (1568, 256), (256, 1), 0), permute_226, out=buf113)
    del permute_226
    buf114 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (256, 1568), (1, 256), 0), view_149, out=buf114)
    del view_149
    buf115 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf116 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf117 = reinterpret_tensor(buf92, (8, 768, 196), (150528, 196, 1), 0); del buf92  # reuse
    cpp_fused_clone_sum_16(c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(getitem_146.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del getitem_146
    buf118 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (196, 6144), (1, 196), 0), view_147, out=buf118)
    del view_147
    buf119 = reinterpret_tensor(buf96, (6144, 196), (196, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (6144, 196), (196, 1), 0), permute_233, out=buf119)
    del permute_233
    buf120 = buf109; del buf109  # reuse
    buf121 = buf108; del buf108  # reuse
    buf122 = empty((768, ), device='cpu', dtype=torch.float32)
    buf123 = empty((768, ), device='cpu', dtype=torch.float32)
    buf124 = buf104; del buf104  # reuse
    buf125 = buf124; del buf124  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_17(c_void_p(buf125.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(mul_197.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(mm_24.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(addmm_48.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del addmm_48
    del div_12
    del mm_24
    del mul_197
    del primals_247
    del primals_250
    buf126 = reinterpret_tensor(buf91, (1568, 256), (256, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (1568, 1536), (1536, 1), 0), permute_236, out=buf126)
    del permute_236
    buf127 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (1536, 1568), (1, 1536), 0), view_145, out=buf127)
    del view_145
    buf128 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf129 = buf121; del buf121  # reuse
    buf130 = buf120; del buf120  # reuse
    buf131 = empty((256, ), device='cpu', dtype=torch.float32)
    buf132 = empty((256, ), device='cpu', dtype=torch.float32)
    buf133 = buf112; del buf112  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_18(c_void_p(buf133.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(mul_192.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del div_13
    del mul_192
    del primals_243
    buf134 = reinterpret_tensor(buf119, (1568, 768), (768, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (1568, 256), (256, 1), 0), permute_240, out=buf134)
    del permute_240
    buf135 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (256, 1568), (1, 256), 0), view_143, out=buf135)
    del view_143
    buf136 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf137 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf138 = reinterpret_tensor(buf113, (8, 768, 196), (150528, 196, 1), 0); del buf113  # reuse
    cpp_fused_clone_sum_19(c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(getitem_140.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    del getitem_140
    buf139 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (196, 6144), (1, 196), 0), view_141, out=buf139)
    del view_141
    buf140 = reinterpret_tensor(buf117, (6144, 196), (196, 1), 0); del buf117  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (6144, 196), (196, 1), 0), permute_247, out=buf140)
    del permute_247
    buf141 = buf130; del buf130  # reuse
    buf142 = buf129; del buf129  # reuse
    buf143 = empty((768, ), device='cpu', dtype=torch.float32)
    buf144 = empty((768, ), device='cpu', dtype=torch.float32)
    buf145 = buf125; del buf125  # reuse
    buf146 = buf145; del buf145  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_20(c_void_p(buf146.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(mul_189.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(mm_23.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    del addmm_46
    del div_14
    del mm_23
    del mul_189
    del primals_237
    del primals_240
    buf147 = buf126; del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (1568, 1536), (1536, 1), 0), permute_250, out=buf147)
    del permute_250
    buf148 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (1536, 1568), (1, 1536), 0), view_139, out=buf148)
    del view_139
    buf149 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf150 = buf142; del buf142  # reuse
    buf151 = buf141; del buf141  # reuse
    buf152 = empty((256, ), device='cpu', dtype=torch.float32)
    buf153 = empty((256, ), device='cpu', dtype=torch.float32)
    buf154 = buf133; del buf133  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_21(c_void_p(buf154.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(mul_184.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del div_15
    del mul_184
    del primals_233
    buf155 = reinterpret_tensor(buf140, (1568, 768), (768, 1), 0); del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (1568, 256), (256, 1), 0), permute_254, out=buf155)
    del permute_254
    buf156 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (256, 1568), (1, 256), 0), view_137, out=buf156)
    del view_137
    buf157 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf158 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf159 = reinterpret_tensor(buf134, (8, 768, 196), (150528, 196, 1), 0); del buf134  # reuse
    cpp_fused_clone_sum_22(c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(getitem_134.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()))
    del getitem_134
    buf160 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (196, 6144), (1, 196), 0), view_135, out=buf160)
    del view_135
    buf161 = reinterpret_tensor(buf138, (6144, 196), (196, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (6144, 196), (196, 1), 0), permute_261, out=buf161)
    del permute_261
    buf162 = buf151; del buf151  # reuse
    buf163 = buf150; del buf150  # reuse
    buf164 = empty((768, ), device='cpu', dtype=torch.float32)
    buf165 = empty((768, ), device='cpu', dtype=torch.float32)
    buf166 = buf146; del buf146  # reuse
    buf167 = buf166; del buf166  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_23(c_void_p(buf167.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(mul_181.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(mm_22.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(addmm_44.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del addmm_44
    del div_16
    del mm_22
    del mul_181
    del primals_227
    del primals_230
    buf168 = buf147; del buf147  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (1568, 1536), (1536, 1), 0), permute_264, out=buf168)
    del permute_264
    buf169 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (1536, 1568), (1, 1536), 0), view_133, out=buf169)
    del view_133
    buf170 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf171 = buf163; del buf163  # reuse
    buf172 = buf162; del buf162  # reuse
    buf173 = empty((256, ), device='cpu', dtype=torch.float32)
    buf174 = empty((256, ), device='cpu', dtype=torch.float32)
    buf175 = buf154; del buf154  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_24(c_void_p(buf175.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(mul_176.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del div_17
    del mul_176
    del primals_223
    buf176 = reinterpret_tensor(buf161, (1568, 768), (768, 1), 0); del buf161  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (1568, 256), (256, 1), 0), permute_268, out=buf176)
    del permute_268
    buf177 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (256, 1568), (1, 256), 0), view_131, out=buf177)
    del view_131
    buf178 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf179 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf180 = reinterpret_tensor(buf155, (8, 768, 196), (150528, 196, 1), 0); del buf155  # reuse
    cpp_fused_clone_sum_25(c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(getitem_128.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()))
    del getitem_128
    buf181 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (196, 6144), (1, 196), 0), view_129, out=buf181)
    del view_129
    buf182 = reinterpret_tensor(buf159, (6144, 196), (196, 1), 0); del buf159  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (6144, 196), (196, 1), 0), permute_275, out=buf182)
    del permute_275
    buf183 = buf172; del buf172  # reuse
    buf184 = buf171; del buf171  # reuse
    buf185 = empty((768, ), device='cpu', dtype=torch.float32)
    buf186 = empty((768, ), device='cpu', dtype=torch.float32)
    buf187 = buf167; del buf167  # reuse
    buf188 = buf187; del buf187  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_26(c_void_p(buf188.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(mul_173.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(mm_21.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del addmm_42
    del div_18
    del mm_21
    del mul_173
    del primals_217
    del primals_220
    buf189 = buf168; del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (1568, 1536), (1536, 1), 0), permute_278, out=buf189)
    del permute_278
    buf190 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (1536, 1568), (1, 1536), 0), view_127, out=buf190)
    del view_127
    buf191 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf192 = buf184; del buf184  # reuse
    buf193 = buf183; del buf183  # reuse
    buf194 = empty((256, ), device='cpu', dtype=torch.float32)
    buf195 = empty((256, ), device='cpu', dtype=torch.float32)
    buf196 = buf175; del buf175  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf196.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(mul_168.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del div_19
    del mul_168
    del primals_213
    buf197 = reinterpret_tensor(buf182, (1568, 768), (768, 1), 0); del buf182  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (1568, 256), (256, 1), 0), permute_282, out=buf197)
    del permute_282
    buf198 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (256, 1568), (1, 256), 0), view_125, out=buf198)
    del view_125
    buf199 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf200 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf201 = reinterpret_tensor(buf176, (8, 768, 196), (150528, 196, 1), 0); del buf176  # reuse
    cpp_fused_clone_sum_28(c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(getitem_122.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del getitem_122
    buf202 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (196, 6144), (1, 196), 0), view_123, out=buf202)
    del view_123
    buf203 = reinterpret_tensor(buf180, (6144, 196), (196, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (6144, 196), (196, 1), 0), permute_289, out=buf203)
    del permute_289
    buf204 = buf193; del buf193  # reuse
    buf205 = buf192; del buf192  # reuse
    buf206 = empty((768, ), device='cpu', dtype=torch.float32)
    buf207 = empty((768, ), device='cpu', dtype=torch.float32)
    buf208 = buf188; del buf188  # reuse
    buf209 = buf208; del buf208  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_29(c_void_p(buf209.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(mul_165.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(mm_20.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del addmm_40
    del div_20
    del mm_20
    del mul_165
    del primals_207
    del primals_210
    buf210 = buf189; del buf189  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (1568, 1536), (1536, 1), 0), permute_292, out=buf210)
    del permute_292
    buf211 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (1536, 1568), (1, 1536), 0), view_121, out=buf211)
    del view_121
    buf212 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf213 = buf205; del buf205  # reuse
    buf214 = buf204; del buf204  # reuse
    buf215 = empty((256, ), device='cpu', dtype=torch.float32)
    buf216 = empty((256, ), device='cpu', dtype=torch.float32)
    buf217 = buf196; del buf196  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_30(c_void_p(buf217.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(mul_160.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del div_21
    del mul_160
    del primals_203
    buf218 = reinterpret_tensor(buf203, (1568, 768), (768, 1), 0); del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (1568, 256), (256, 1), 0), permute_296, out=buf218)
    del permute_296
    buf219 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (256, 1568), (1, 256), 0), view_119, out=buf219)
    del view_119
    buf220 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf221 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf222 = reinterpret_tensor(buf197, (8, 768, 196), (150528, 196, 1), 0); del buf197  # reuse
    cpp_fused_clone_sum_31(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(getitem_116.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del getitem_116
    buf223 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (196, 6144), (1, 196), 0), view_117, out=buf223)
    del view_117
    buf224 = reinterpret_tensor(buf201, (6144, 196), (196, 1), 0); del buf201  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (6144, 196), (196, 1), 0), permute_303, out=buf224)
    del permute_303
    buf225 = buf214; del buf214  # reuse
    buf226 = buf213; del buf213  # reuse
    buf227 = empty((768, ), device='cpu', dtype=torch.float32)
    buf228 = empty((768, ), device='cpu', dtype=torch.float32)
    buf229 = buf209; del buf209  # reuse
    buf230 = buf229; del buf229  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_32(c_void_p(buf230.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(mul_157.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(mm_19.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(addmm_38.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del addmm_38
    del div_22
    del mm_19
    del mul_157
    del primals_197
    del primals_200
    buf231 = buf210; del buf210  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (1568, 1536), (1536, 1), 0), permute_306, out=buf231)
    del permute_306
    buf232 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (1536, 1568), (1, 1536), 0), view_115, out=buf232)
    del view_115
    buf233 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf234 = buf226; del buf226  # reuse
    buf235 = buf225; del buf225  # reuse
    buf236 = empty((256, ), device='cpu', dtype=torch.float32)
    buf237 = empty((256, ), device='cpu', dtype=torch.float32)
    buf238 = buf217; del buf217  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_33(c_void_p(buf238.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(mul_152.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    del div_23
    del mul_152
    del primals_193
    buf239 = reinterpret_tensor(buf224, (1568, 768), (768, 1), 0); del buf224  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf238, (1568, 256), (256, 1), 0), permute_310, out=buf239)
    del permute_310
    buf240 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf238, (256, 1568), (1, 256), 0), view_113, out=buf240)
    del view_113
    buf241 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf242 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf243 = reinterpret_tensor(buf218, (8, 768, 196), (150528, 196, 1), 0); del buf218  # reuse
    cpp_fused_clone_sum_34(c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(getitem_110.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    del getitem_110
    buf244 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (196, 6144), (1, 196), 0), view_111, out=buf244)
    del view_111
    buf245 = reinterpret_tensor(buf222, (6144, 196), (196, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (6144, 196), (196, 1), 0), permute_317, out=buf245)
    del permute_317
    buf246 = buf235; del buf235  # reuse
    buf247 = buf234; del buf234  # reuse
    buf248 = empty((768, ), device='cpu', dtype=torch.float32)
    buf249 = empty((768, ), device='cpu', dtype=torch.float32)
    buf250 = buf230; del buf230  # reuse
    buf251 = buf250; del buf250  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_35(c_void_p(buf251.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(mul_149.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(mm_18.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(addmm_36.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    del addmm_36
    del div_24
    del mm_18
    del mul_149
    del primals_187
    del primals_190
    buf252 = buf231; del buf231  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (1568, 1536), (1536, 1), 0), permute_320, out=buf252)
    del permute_320
    buf253 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (1536, 1568), (1, 1536), 0), view_109, out=buf253)
    del view_109
    buf254 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf255 = buf247; del buf247  # reuse
    buf256 = buf246; del buf246  # reuse
    buf257 = empty((256, ), device='cpu', dtype=torch.float32)
    buf258 = empty((256, ), device='cpu', dtype=torch.float32)
    buf259 = buf238; del buf238  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_36(c_void_p(buf259.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(mul_144.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del div_25
    del mul_144
    del primals_183
    buf260 = reinterpret_tensor(buf245, (1568, 768), (768, 1), 0); del buf245  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (1568, 256), (256, 1), 0), permute_324, out=buf260)
    del permute_324
    buf261 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (256, 1568), (1, 256), 0), view_107, out=buf261)
    del view_107
    buf262 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf263 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf264 = reinterpret_tensor(buf239, (8, 768, 196), (150528, 196, 1), 0); del buf239  # reuse
    cpp_fused_clone_sum_37(c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(getitem_104.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    del getitem_104
    buf265 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf264, (196, 6144), (1, 196), 0), view_105, out=buf265)
    del view_105
    buf266 = reinterpret_tensor(buf243, (6144, 196), (196, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf264, (6144, 196), (196, 1), 0), permute_331, out=buf266)
    del permute_331
    buf267 = buf256; del buf256  # reuse
    buf268 = buf255; del buf255  # reuse
    buf269 = empty((768, ), device='cpu', dtype=torch.float32)
    buf270 = empty((768, ), device='cpu', dtype=torch.float32)
    buf271 = buf251; del buf251  # reuse
    buf272 = buf271; del buf271  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_38(c_void_p(buf272.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(mul_141.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(mm_17.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del addmm_34
    del div_26
    del mm_17
    del mul_141
    del primals_177
    del primals_180
    buf273 = buf252; del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (1568, 1536), (1536, 1), 0), permute_334, out=buf273)
    del permute_334
    buf274 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (1536, 1568), (1, 1536), 0), view_103, out=buf274)
    del view_103
    buf275 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf276 = buf268; del buf268  # reuse
    buf277 = buf267; del buf267  # reuse
    buf278 = empty((256, ), device='cpu', dtype=torch.float32)
    buf279 = empty((256, ), device='cpu', dtype=torch.float32)
    buf280 = buf259; del buf259  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_39(c_void_p(buf280.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(mul_136.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del div_27
    del mul_136
    del primals_173
    buf281 = reinterpret_tensor(buf266, (1568, 768), (768, 1), 0); del buf266  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (1568, 256), (256, 1), 0), permute_338, out=buf281)
    del permute_338
    buf282 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (256, 1568), (1, 256), 0), view_101, out=buf282)
    del view_101
    buf283 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf284 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf285 = reinterpret_tensor(buf260, (8, 768, 196), (150528, 196, 1), 0); del buf260  # reuse
    cpp_fused_clone_sum_40(c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(getitem_98.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del getitem_98
    buf286 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf285, (196, 6144), (1, 196), 0), view_99, out=buf286)
    del view_99
    buf287 = reinterpret_tensor(buf264, (6144, 196), (196, 1), 0); del buf264  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf285, (6144, 196), (196, 1), 0), permute_345, out=buf287)
    del permute_345
    buf288 = buf277; del buf277  # reuse
    buf289 = buf276; del buf276  # reuse
    buf290 = empty((768, ), device='cpu', dtype=torch.float32)
    buf291 = empty((768, ), device='cpu', dtype=torch.float32)
    buf292 = buf272; del buf272  # reuse
    buf293 = buf292; del buf292  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_41(c_void_p(buf293.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(mul_133.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(mm_16.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(addmm_32.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del addmm_32
    del div_28
    del mm_16
    del mul_133
    del primals_167
    del primals_170
    buf294 = buf273; del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf293, (1568, 1536), (1536, 1), 0), permute_348, out=buf294)
    del permute_348
    buf295 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf293, (1536, 1568), (1, 1536), 0), view_97, out=buf295)
    del view_97
    buf296 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf297 = buf289; del buf289  # reuse
    buf298 = buf288; del buf288  # reuse
    buf299 = empty((256, ), device='cpu', dtype=torch.float32)
    buf300 = empty((256, ), device='cpu', dtype=torch.float32)
    buf301 = buf280; del buf280  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_42(c_void_p(buf301.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(mul_128.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del div_29
    del mul_128
    del primals_163
    buf302 = reinterpret_tensor(buf287, (1568, 768), (768, 1), 0); del buf287  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf301, (1568, 256), (256, 1), 0), permute_352, out=buf302)
    del permute_352
    buf303 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf301, (256, 1568), (1, 256), 0), view_95, out=buf303)
    del view_95
    buf304 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf305 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf306 = reinterpret_tensor(buf281, (8, 768, 196), (150528, 196, 1), 0); del buf281  # reuse
    cpp_fused_clone_sum_43(c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(getitem_92.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del getitem_92
    buf307 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (196, 6144), (1, 196), 0), view_93, out=buf307)
    del view_93
    buf308 = reinterpret_tensor(buf285, (6144, 196), (196, 1), 0); del buf285  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (6144, 196), (196, 1), 0), permute_359, out=buf308)
    del permute_359
    buf309 = buf298; del buf298  # reuse
    buf310 = buf297; del buf297  # reuse
    buf311 = empty((768, ), device='cpu', dtype=torch.float32)
    buf312 = empty((768, ), device='cpu', dtype=torch.float32)
    buf313 = buf293; del buf293  # reuse
    buf314 = buf313; del buf313  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_44(c_void_p(buf314.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(mul_125.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(mm_15.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del addmm_30
    del div_30
    del mm_15
    del mul_125
    del primals_157
    del primals_160
    buf315 = buf294; del buf294  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf314, (1568, 1536), (1536, 1), 0), permute_362, out=buf315)
    del permute_362
    buf316 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf314, (1536, 1568), (1, 1536), 0), view_91, out=buf316)
    del view_91
    buf317 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf318 = buf310; del buf310  # reuse
    buf319 = buf309; del buf309  # reuse
    buf320 = empty((256, ), device='cpu', dtype=torch.float32)
    buf321 = empty((256, ), device='cpu', dtype=torch.float32)
    buf322 = buf301; del buf301  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_45(c_void_p(buf322.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(mul_120.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del div_31
    del mul_120
    del primals_153
    buf323 = reinterpret_tensor(buf308, (1568, 768), (768, 1), 0); del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (1568, 256), (256, 1), 0), permute_366, out=buf323)
    del permute_366
    buf324 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (256, 1568), (1, 256), 0), view_89, out=buf324)
    del view_89
    buf325 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf326 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf327 = reinterpret_tensor(buf302, (8, 768, 196), (150528, 196, 1), 0); del buf302  # reuse
    cpp_fused_clone_sum_46(c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(getitem_86.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del getitem_86
    buf328 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (196, 6144), (1, 196), 0), view_87, out=buf328)
    del view_87
    buf329 = reinterpret_tensor(buf306, (6144, 196), (196, 1), 0); del buf306  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (6144, 196), (196, 1), 0), permute_373, out=buf329)
    del permute_373
    buf330 = buf319; del buf319  # reuse
    buf331 = buf318; del buf318  # reuse
    buf332 = empty((768, ), device='cpu', dtype=torch.float32)
    buf333 = empty((768, ), device='cpu', dtype=torch.float32)
    buf334 = buf314; del buf314  # reuse
    buf335 = buf334; del buf334  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_47(c_void_p(buf335.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(mul_117.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(mm_14.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del addmm_28
    del div_32
    del mm_14
    del mul_117
    del primals_147
    del primals_150
    buf336 = buf315; del buf315  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (1568, 1536), (1536, 1), 0), permute_376, out=buf336)
    del permute_376
    buf337 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (1536, 1568), (1, 1536), 0), view_85, out=buf337)
    del view_85
    buf338 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf339 = buf331; del buf331  # reuse
    buf340 = buf330; del buf330  # reuse
    buf341 = empty((256, ), device='cpu', dtype=torch.float32)
    buf342 = empty((256, ), device='cpu', dtype=torch.float32)
    buf343 = buf322; del buf322  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_48(c_void_p(buf343.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(mul_112.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del div_33
    del mul_112
    del primals_143
    buf344 = reinterpret_tensor(buf329, (1568, 768), (768, 1), 0); del buf329  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (1568, 256), (256, 1), 0), permute_380, out=buf344)
    del permute_380
    buf345 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (256, 1568), (1, 256), 0), view_83, out=buf345)
    del view_83
    buf346 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf347 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf348 = reinterpret_tensor(buf323, (8, 768, 196), (150528, 196, 1), 0); del buf323  # reuse
    cpp_fused_clone_sum_49(c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(getitem_80.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    del getitem_80
    buf349 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf348, (196, 6144), (1, 196), 0), view_81, out=buf349)
    del view_81
    buf350 = reinterpret_tensor(buf327, (6144, 196), (196, 1), 0); del buf327  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf348, (6144, 196), (196, 1), 0), permute_387, out=buf350)
    del permute_387
    buf351 = buf340; del buf340  # reuse
    buf352 = buf339; del buf339  # reuse
    buf353 = empty((768, ), device='cpu', dtype=torch.float32)
    buf354 = empty((768, ), device='cpu', dtype=torch.float32)
    buf355 = buf335; del buf335  # reuse
    buf356 = buf355; del buf355  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_50(c_void_p(buf356.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(mul_109.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(mm_13.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    del addmm_26
    del div_34
    del mm_13
    del mul_109
    del primals_137
    del primals_140
    buf357 = buf336; del buf336  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf356, (1568, 1536), (1536, 1), 0), permute_390, out=buf357)
    del permute_390
    buf358 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf356, (1536, 1568), (1, 1536), 0), view_79, out=buf358)
    del view_79
    buf359 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf360 = buf352; del buf352  # reuse
    buf361 = buf351; del buf351  # reuse
    buf362 = empty((256, ), device='cpu', dtype=torch.float32)
    buf363 = empty((256, ), device='cpu', dtype=torch.float32)
    buf364 = buf343; del buf343  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_51(c_void_p(buf364.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(mul_104.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    del div_35
    del mul_104
    del primals_133
    buf365 = reinterpret_tensor(buf350, (1568, 768), (768, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (1568, 256), (256, 1), 0), permute_394, out=buf365)
    del permute_394
    buf366 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (256, 1568), (1, 256), 0), view_77, out=buf366)
    del view_77
    buf367 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf368 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf369 = reinterpret_tensor(buf344, (8, 768, 196), (150528, 196, 1), 0); del buf344  # reuse
    cpp_fused_clone_sum_52(c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(getitem_74.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()))
    del getitem_74
    buf370 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf369, (196, 6144), (1, 196), 0), view_75, out=buf370)
    del view_75
    buf371 = reinterpret_tensor(buf348, (6144, 196), (196, 1), 0); del buf348  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf369, (6144, 196), (196, 1), 0), permute_401, out=buf371)
    del permute_401
    buf372 = buf361; del buf361  # reuse
    buf373 = buf360; del buf360  # reuse
    buf374 = empty((768, ), device='cpu', dtype=torch.float32)
    buf375 = empty((768, ), device='cpu', dtype=torch.float32)
    buf376 = buf356; del buf356  # reuse
    buf377 = buf376; del buf376  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_53(c_void_p(buf377.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(mul_101.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(mm_12.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(addmm_24.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    del addmm_24
    del div_36
    del mm_12
    del mul_101
    del primals_127
    del primals_130
    buf378 = buf357; del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (1568, 1536), (1536, 1), 0), permute_404, out=buf378)
    del permute_404
    buf379 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (1536, 1568), (1, 1536), 0), view_73, out=buf379)
    del view_73
    buf380 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf381 = buf373; del buf373  # reuse
    buf382 = buf372; del buf372  # reuse
    buf383 = empty((256, ), device='cpu', dtype=torch.float32)
    buf384 = empty((256, ), device='cpu', dtype=torch.float32)
    buf385 = buf364; del buf364  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_54(c_void_p(buf385.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del div_37
    del mul_96
    del primals_123
    buf386 = reinterpret_tensor(buf371, (1568, 768), (768, 1), 0); del buf371  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (1568, 256), (256, 1), 0), permute_408, out=buf386)
    del permute_408
    buf387 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (256, 1568), (1, 256), 0), view_71, out=buf387)
    del view_71
    buf388 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf389 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf390 = reinterpret_tensor(buf365, (8, 768, 196), (150528, 196, 1), 0); del buf365  # reuse
    cpp_fused_clone_sum_55(c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(getitem_68.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()))
    del getitem_68
    buf391 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (196, 6144), (1, 196), 0), view_69, out=buf391)
    del view_69
    buf392 = reinterpret_tensor(buf369, (6144, 196), (196, 1), 0); del buf369  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (6144, 196), (196, 1), 0), permute_415, out=buf392)
    del permute_415
    buf393 = buf382; del buf382  # reuse
    buf394 = buf381; del buf381  # reuse
    buf395 = empty((768, ), device='cpu', dtype=torch.float32)
    buf396 = empty((768, ), device='cpu', dtype=torch.float32)
    buf397 = buf377; del buf377  # reuse
    buf398 = buf397; del buf397  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_56(c_void_p(buf398.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(mul_93.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(mm_11.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(div_38.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del addmm_22
    del div_38
    del mm_11
    del mul_93
    del primals_117
    del primals_120
    buf399 = buf378; del buf378  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf398, (1568, 1536), (1536, 1), 0), permute_418, out=buf399)
    del permute_418
    buf400 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf398, (1536, 1568), (1, 1536), 0), view_67, out=buf400)
    del view_67
    buf401 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf402 = buf394; del buf394  # reuse
    buf403 = buf393; del buf393  # reuse
    buf404 = empty((256, ), device='cpu', dtype=torch.float32)
    buf405 = empty((256, ), device='cpu', dtype=torch.float32)
    buf406 = buf385; del buf385  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_57(c_void_p(buf406.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    del div_39
    del mul_88
    del primals_113
    buf407 = reinterpret_tensor(buf392, (1568, 768), (768, 1), 0); del buf392  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf406, (1568, 256), (256, 1), 0), permute_422, out=buf407)
    del permute_422
    buf408 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf406, (256, 1568), (1, 256), 0), view_65, out=buf408)
    del view_65
    buf409 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf410 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf411 = reinterpret_tensor(buf386, (8, 768, 196), (150528, 196, 1), 0); del buf386  # reuse
    cpp_fused_clone_sum_58(c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(getitem_62.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    del getitem_62
    buf412 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf411, (196, 6144), (1, 196), 0), view_63, out=buf412)
    del view_63
    buf413 = reinterpret_tensor(buf390, (6144, 196), (196, 1), 0); del buf390  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf411, (6144, 196), (196, 1), 0), permute_429, out=buf413)
    del permute_429
    buf414 = buf403; del buf403  # reuse
    buf415 = buf402; del buf402  # reuse
    buf416 = empty((768, ), device='cpu', dtype=torch.float32)
    buf417 = empty((768, ), device='cpu', dtype=torch.float32)
    buf418 = buf398; del buf398  # reuse
    buf419 = buf418; del buf418  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_59(c_void_p(buf419.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(mul_85.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(mm_10.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(addmm_20.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    del addmm_20
    del div_40
    del mm_10
    del mul_85
    del primals_107
    del primals_110
    buf420 = buf399; del buf399  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (1568, 1536), (1536, 1), 0), permute_432, out=buf420)
    del permute_432
    buf421 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (1536, 1568), (1, 1536), 0), view_61, out=buf421)
    del view_61
    buf422 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf423 = buf415; del buf415  # reuse
    buf424 = buf414; del buf414  # reuse
    buf425 = empty((256, ), device='cpu', dtype=torch.float32)
    buf426 = empty((256, ), device='cpu', dtype=torch.float32)
    buf427 = buf406; del buf406  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_60(c_void_p(buf427.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    del div_41
    del mul_80
    del primals_103
    buf428 = reinterpret_tensor(buf413, (1568, 768), (768, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (1568, 256), (256, 1), 0), permute_436, out=buf428)
    del permute_436
    buf429 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (256, 1568), (1, 256), 0), view_59, out=buf429)
    del view_59
    buf430 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf431 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf432 = reinterpret_tensor(buf407, (8, 768, 196), (150528, 196, 1), 0); del buf407  # reuse
    cpp_fused_clone_sum_61(c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(getitem_56.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()))
    del getitem_56
    buf433 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (196, 6144), (1, 196), 0), view_57, out=buf433)
    del view_57
    buf434 = reinterpret_tensor(buf411, (6144, 196), (196, 1), 0); del buf411  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (6144, 196), (196, 1), 0), permute_443, out=buf434)
    del permute_443
    buf435 = buf424; del buf424  # reuse
    buf436 = buf423; del buf423  # reuse
    buf437 = empty((768, ), device='cpu', dtype=torch.float32)
    buf438 = empty((768, ), device='cpu', dtype=torch.float32)
    buf439 = buf419; del buf419  # reuse
    buf440 = buf439; del buf439  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_62(c_void_p(buf440.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(mul_77.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(mm_9.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    del addmm_18
    del div_42
    del mm_9
    del mul_77
    del primals_100
    del primals_97
    buf441 = buf420; del buf420  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (1568, 1536), (1536, 1), 0), permute_446, out=buf441)
    del permute_446
    buf442 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (1536, 1568), (1, 1536), 0), view_55, out=buf442)
    del view_55
    buf443 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf444 = buf436; del buf436  # reuse
    buf445 = buf435; del buf435  # reuse
    buf446 = empty((256, ), device='cpu', dtype=torch.float32)
    buf447 = empty((256, ), device='cpu', dtype=torch.float32)
    buf448 = buf427; del buf427  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_63(c_void_p(buf448.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    del div_43
    del mul_72
    del primals_93
    buf449 = reinterpret_tensor(buf434, (1568, 768), (768, 1), 0); del buf434  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (1568, 256), (256, 1), 0), permute_450, out=buf449)
    del permute_450
    buf450 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (256, 1568), (1, 256), 0), view_53, out=buf450)
    del view_53
    buf451 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf452 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf453 = reinterpret_tensor(buf428, (8, 768, 196), (150528, 196, 1), 0); del buf428  # reuse
    cpp_fused_clone_sum_64(c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(getitem_50.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    del getitem_50
    buf454 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (196, 6144), (1, 196), 0), view_51, out=buf454)
    del view_51
    buf455 = reinterpret_tensor(buf432, (6144, 196), (196, 1), 0); del buf432  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (6144, 196), (196, 1), 0), permute_457, out=buf455)
    del permute_457
    buf456 = buf445; del buf445  # reuse
    buf457 = buf444; del buf444  # reuse
    buf458 = empty((768, ), device='cpu', dtype=torch.float32)
    buf459 = empty((768, ), device='cpu', dtype=torch.float32)
    buf460 = buf440; del buf440  # reuse
    buf461 = buf460; del buf460  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_65(c_void_p(buf461.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(mul_69.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(mm_8.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(div_44.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()))
    del addmm_16
    del div_44
    del mm_8
    del mul_69
    del primals_87
    del primals_90
    buf462 = buf441; del buf441  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (1568, 1536), (1536, 1), 0), permute_460, out=buf462)
    del permute_460
    buf463 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (1536, 1568), (1, 1536), 0), view_49, out=buf463)
    del view_49
    buf464 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf465 = buf457; del buf457  # reuse
    buf466 = buf456; del buf456  # reuse
    buf467 = empty((256, ), device='cpu', dtype=torch.float32)
    buf468 = empty((256, ), device='cpu', dtype=torch.float32)
    buf469 = buf448; del buf448  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_66(c_void_p(buf469.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    del div_45
    del mul_64
    del primals_83
    buf470 = reinterpret_tensor(buf455, (1568, 768), (768, 1), 0); del buf455  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf469, (1568, 256), (256, 1), 0), permute_464, out=buf470)
    del permute_464
    buf471 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf469, (256, 1568), (1, 256), 0), view_47, out=buf471)
    del view_47
    buf472 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf473 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf474 = reinterpret_tensor(buf449, (8, 768, 196), (150528, 196, 1), 0); del buf449  # reuse
    cpp_fused_clone_sum_67(c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(getitem_44.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()))
    del getitem_44
    buf475 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (196, 6144), (1, 196), 0), view_45, out=buf475)
    del view_45
    buf476 = reinterpret_tensor(buf453, (6144, 196), (196, 1), 0); del buf453  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (6144, 196), (196, 1), 0), permute_471, out=buf476)
    del permute_471
    buf477 = buf466; del buf466  # reuse
    buf478 = buf465; del buf465  # reuse
    buf479 = empty((768, ), device='cpu', dtype=torch.float32)
    buf480 = empty((768, ), device='cpu', dtype=torch.float32)
    buf481 = buf461; del buf461  # reuse
    buf482 = buf481; del buf481  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_68(c_void_p(buf482.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(mul_61.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(mm_7.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()))
    del addmm_14
    del div_46
    del mm_7
    del mul_61
    del primals_77
    del primals_80
    buf483 = buf462; del buf462  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (1568, 1536), (1536, 1), 0), permute_474, out=buf483)
    del permute_474
    buf484 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (1536, 1568), (1, 1536), 0), view_43, out=buf484)
    del view_43
    buf485 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf486 = buf478; del buf478  # reuse
    buf487 = buf477; del buf477  # reuse
    buf488 = empty((256, ), device='cpu', dtype=torch.float32)
    buf489 = empty((256, ), device='cpu', dtype=torch.float32)
    buf490 = buf469; del buf469  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_69(c_void_p(buf490.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_47.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()))
    del div_47
    del mul_56
    del primals_73
    buf491 = reinterpret_tensor(buf476, (1568, 768), (768, 1), 0); del buf476  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf490, (1568, 256), (256, 1), 0), permute_478, out=buf491)
    del permute_478
    buf492 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf490, (256, 1568), (1, 256), 0), view_41, out=buf492)
    del view_41
    buf493 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf494 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf495 = reinterpret_tensor(buf470, (8, 768, 196), (150528, 196, 1), 0); del buf470  # reuse
    cpp_fused_clone_sum_70(c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(getitem_38.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    del getitem_38
    buf496 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (196, 6144), (1, 196), 0), view_39, out=buf496)
    del view_39
    buf497 = reinterpret_tensor(buf474, (6144, 196), (196, 1), 0); del buf474  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (6144, 196), (196, 1), 0), permute_485, out=buf497)
    del permute_485
    buf498 = buf487; del buf487  # reuse
    buf499 = buf486; del buf486  # reuse
    buf500 = empty((768, ), device='cpu', dtype=torch.float32)
    buf501 = empty((768, ), device='cpu', dtype=torch.float32)
    buf502 = buf482; del buf482  # reuse
    buf503 = buf502; del buf502  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_71(c_void_p(buf503.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(mul_53.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(mm_6.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(addmm_12.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    del addmm_12
    del div_48
    del mm_6
    del mul_53
    del primals_67
    del primals_70
    buf504 = buf483; del buf483  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (1568, 1536), (1536, 1), 0), permute_488, out=buf504)
    del permute_488
    buf505 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (1536, 1568), (1, 1536), 0), view_37, out=buf505)
    del view_37
    buf506 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf507 = buf499; del buf499  # reuse
    buf508 = buf498; del buf498  # reuse
    buf509 = empty((256, ), device='cpu', dtype=torch.float32)
    buf510 = empty((256, ), device='cpu', dtype=torch.float32)
    buf511 = buf490; del buf490  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_72(c_void_p(buf511.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()))
    del div_49
    del mul_48
    del primals_63
    buf512 = reinterpret_tensor(buf497, (1568, 768), (768, 1), 0); del buf497  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf511, (1568, 256), (256, 1), 0), permute_492, out=buf512)
    del permute_492
    buf513 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf511, (256, 1568), (1, 256), 0), view_35, out=buf513)
    del view_35
    buf514 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf515 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf516 = reinterpret_tensor(buf491, (8, 768, 196), (150528, 196, 1), 0); del buf491  # reuse
    cpp_fused_clone_sum_73(c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(getitem_32.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    del getitem_32
    buf517 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf516, (196, 6144), (1, 196), 0), view_33, out=buf517)
    del view_33
    buf518 = reinterpret_tensor(buf495, (6144, 196), (196, 1), 0); del buf495  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf516, (6144, 196), (196, 1), 0), permute_499, out=buf518)
    del permute_499
    buf519 = buf508; del buf508  # reuse
    buf520 = buf507; del buf507  # reuse
    buf521 = empty((768, ), device='cpu', dtype=torch.float32)
    buf522 = empty((768, ), device='cpu', dtype=torch.float32)
    buf523 = buf503; del buf503  # reuse
    buf524 = buf523; del buf523  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_74(c_void_p(buf524.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(mul_45.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(div_50.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()))
    del addmm_10
    del div_50
    del mm_5
    del mul_45
    del primals_57
    del primals_60
    buf525 = buf504; del buf504  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (1568, 1536), (1536, 1), 0), permute_502, out=buf525)
    del permute_502
    buf526 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (1536, 1568), (1, 1536), 0), view_31, out=buf526)
    del view_31
    buf527 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf528 = buf520; del buf520  # reuse
    buf529 = buf519; del buf519  # reuse
    buf530 = empty((256, ), device='cpu', dtype=torch.float32)
    buf531 = empty((256, ), device='cpu', dtype=torch.float32)
    buf532 = buf511; del buf511  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_75(c_void_p(buf532.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()))
    del div_51
    del mul_40
    del primals_53
    buf533 = reinterpret_tensor(buf518, (1568, 768), (768, 1), 0); del buf518  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf532, (1568, 256), (256, 1), 0), permute_506, out=buf533)
    del permute_506
    buf534 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf532, (256, 1568), (1, 256), 0), view_29, out=buf534)
    del view_29
    buf535 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf536 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf537 = reinterpret_tensor(buf512, (8, 768, 196), (150528, 196, 1), 0); del buf512  # reuse
    cpp_fused_clone_sum_76(c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(getitem_26.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()))
    del getitem_26
    buf538 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (196, 6144), (1, 196), 0), view_27, out=buf538)
    del view_27
    buf539 = reinterpret_tensor(buf516, (6144, 196), (196, 1), 0); del buf516  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (6144, 196), (196, 1), 0), permute_513, out=buf539)
    del permute_513
    buf540 = buf529; del buf529  # reuse
    buf541 = buf528; del buf528  # reuse
    buf542 = empty((768, ), device='cpu', dtype=torch.float32)
    buf543 = empty((768, ), device='cpu', dtype=torch.float32)
    buf544 = buf524; del buf524  # reuse
    buf545 = buf544; del buf544  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_77(c_void_p(buf545.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(mul_37.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(mm_4.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(addmm_8.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()))
    del addmm_8
    del div_52
    del mm_4
    del mul_37
    del primals_47
    del primals_50
    buf546 = buf525; del buf525  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf545, (1568, 1536), (1536, 1), 0), permute_516, out=buf546)
    del permute_516
    buf547 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf545, (1536, 1568), (1, 1536), 0), view_25, out=buf547)
    del view_25
    buf548 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf549 = buf541; del buf541  # reuse
    buf550 = buf540; del buf540  # reuse
    buf551 = empty((256, ), device='cpu', dtype=torch.float32)
    buf552 = empty((256, ), device='cpu', dtype=torch.float32)
    buf553 = buf532; del buf532  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_78(c_void_p(buf553.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_53.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()))
    del div_53
    del mul_32
    del primals_43
    buf554 = reinterpret_tensor(buf539, (1568, 768), (768, 1), 0); del buf539  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf553, (1568, 256), (256, 1), 0), permute_520, out=buf554)
    del permute_520
    buf555 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf553, (256, 1568), (1, 256), 0), view_23, out=buf555)
    del view_23
    buf556 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf557 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf558 = reinterpret_tensor(buf533, (8, 768, 196), (150528, 196, 1), 0); del buf533  # reuse
    cpp_fused_clone_sum_79(c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(getitem_20.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()))
    del getitem_20
    buf559 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf558, (196, 6144), (1, 196), 0), view_21, out=buf559)
    del view_21
    buf560 = reinterpret_tensor(buf537, (6144, 196), (196, 1), 0); del buf537  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf558, (6144, 196), (196, 1), 0), permute_527, out=buf560)
    del permute_527
    buf561 = buf550; del buf550  # reuse
    buf562 = buf549; del buf549  # reuse
    buf563 = empty((768, ), device='cpu', dtype=torch.float32)
    buf564 = empty((768, ), device='cpu', dtype=torch.float32)
    buf565 = buf545; del buf545  # reuse
    buf566 = buf565; del buf565  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_80(c_void_p(buf566.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(mul_29.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()))
    del addmm_6
    del div_54
    del mm_3
    del mul_29
    del primals_37
    del primals_40
    buf567 = buf546; del buf546  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf566, (1568, 1536), (1536, 1), 0), permute_530, out=buf567)
    del permute_530
    buf568 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf566, (1536, 1568), (1, 1536), 0), view_19, out=buf568)
    del view_19
    buf569 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf570 = buf562; del buf562  # reuse
    buf571 = buf561; del buf561  # reuse
    buf572 = empty((256, ), device='cpu', dtype=torch.float32)
    buf573 = empty((256, ), device='cpu', dtype=torch.float32)
    buf574 = buf553; del buf553  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_81(c_void_p(buf574.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()))
    del div_55
    del mul_24
    del primals_33
    buf575 = reinterpret_tensor(buf560, (1568, 768), (768, 1), 0); del buf560  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf574, (1568, 256), (256, 1), 0), permute_534, out=buf575)
    del permute_534
    buf576 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf574, (256, 1568), (1, 256), 0), view_17, out=buf576)
    del view_17
    buf577 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf578 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf579 = reinterpret_tensor(buf554, (8, 768, 196), (150528, 196, 1), 0); del buf554  # reuse
    cpp_fused_clone_sum_82(c_void_p(buf574.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(getitem_14.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()))
    del getitem_14
    buf580 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf579, (196, 6144), (1, 196), 0), view_15, out=buf580)
    del view_15
    buf581 = reinterpret_tensor(buf558, (6144, 196), (196, 1), 0); del buf558  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf579, (6144, 196), (196, 1), 0), permute_541, out=buf581)
    del permute_541
    buf582 = buf571; del buf571  # reuse
    buf583 = buf570; del buf570  # reuse
    buf584 = empty((768, ), device='cpu', dtype=torch.float32)
    buf585 = empty((768, ), device='cpu', dtype=torch.float32)
    buf586 = buf566; del buf566  # reuse
    buf587 = buf586; del buf586  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_83(c_void_p(buf587.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(mul_21.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(mm_2.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(div_56.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()))
    del addmm_4
    del div_56
    del mm_2
    del mul_21
    del primals_27
    del primals_30
    buf588 = buf567; del buf567  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf587, (1568, 1536), (1536, 1), 0), permute_544, out=buf588)
    del permute_544
    buf589 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf587, (1536, 1568), (1, 1536), 0), view_13, out=buf589)
    del view_13
    buf590 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf591 = buf583; del buf583  # reuse
    buf592 = buf582; del buf582  # reuse
    buf593 = empty((256, ), device='cpu', dtype=torch.float32)
    buf594 = empty((256, ), device='cpu', dtype=torch.float32)
    buf595 = buf574; del buf574  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_84(c_void_p(buf595.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()))
    del div_57
    del mul_16
    del primals_23
    buf596 = reinterpret_tensor(buf581, (1568, 768), (768, 1), 0); del buf581  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf595, (1568, 256), (256, 1), 0), permute_548, out=buf596)
    del permute_548
    buf597 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf595, (256, 1568), (1, 256), 0), view_11, out=buf597)
    del view_11
    buf598 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf599 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf600 = reinterpret_tensor(buf575, (8, 768, 196), (150528, 196, 1), 0); del buf575  # reuse
    cpp_fused_clone_sum_85(c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(getitem_8.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()))
    del getitem_8
    buf601 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf600, (196, 6144), (1, 196), 0), view_9, out=buf601)
    del view_9
    buf602 = reinterpret_tensor(buf579, (6144, 196), (196, 1), 0); del buf579  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf600, (6144, 196), (196, 1), 0), permute_555, out=buf602)
    del permute_555
    buf603 = buf592; del buf592  # reuse
    buf604 = buf591; del buf591  # reuse
    buf605 = empty((768, ), device='cpu', dtype=torch.float32)
    buf606 = empty((768, ), device='cpu', dtype=torch.float32)
    buf607 = buf587; del buf587  # reuse
    buf608 = buf607; del buf607  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_86(c_void_p(buf608.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(mul_13.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(mm_1.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    del addmm_2
    del div_58
    del mm_1
    del mul_13
    del primals_17
    del primals_20
    buf609 = buf588; del buf588  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf608, (1568, 1536), (1536, 1), 0), permute_558, out=buf609)
    del permute_558
    buf610 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf608, (1536, 1568), (1, 1536), 0), view_7, out=buf610)
    del view_7
    buf611 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf612 = buf604; del buf604  # reuse
    buf613 = buf603; del buf603  # reuse
    buf614 = empty((256, ), device='cpu', dtype=torch.float32)
    buf615 = empty((256, ), device='cpu', dtype=torch.float32)
    buf616 = buf595; del buf595  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_87(c_void_p(buf616.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_59.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()))
    del div_59
    del mul_8
    del primals_13
    buf617 = reinterpret_tensor(buf602, (1568, 768), (768, 1), 0); del buf602  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf616, (1568, 256), (256, 1), 0), permute_562, out=buf617)
    del permute_562
    buf618 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf616, (256, 1568), (1, 256), 0), view_5, out=buf618)
    del view_5
    buf619 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf620 = empty((1, 1, 196), device='cpu', dtype=torch.float32)
    buf621 = reinterpret_tensor(buf596, (8, 768, 196), (150528, 196, 1), 0); del buf596  # reuse
    cpp_fused_clone_sum_88(c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(getitem_2.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()))
    del getitem_2
    buf622 = empty((196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf621, (196, 6144), (1, 196), 0), view_3, out=buf622)
    del view_3
    buf623 = reinterpret_tensor(buf600, (6144, 196), (196, 1), 0); del buf600  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf621, (6144, 196), (196, 1), 0), permute_569, out=buf623)
    del buf621
    del permute_569
    buf624 = buf613; del buf613  # reuse
    buf625 = buf612; del buf612  # reuse
    buf626 = empty((768, ), device='cpu', dtype=torch.float32)
    buf627 = empty((768, ), device='cpu', dtype=torch.float32)
    buf628 = buf608; del buf608  # reuse
    buf629 = buf628; del buf628  # reuse
    cpp_fused_cat_gelu_gelu_backward_native_layer_norm_backward_89(c_void_p(buf629.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(mul_5.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(mm.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(addmm.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()))
    del addmm
    del buf617
    del buf623
    del div_60
    del mm
    del mul_5
    del primals_10
    del primals_7
    buf630 = buf609; del buf609  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (1568, 1536), (1536, 1), 0), permute_572, out=buf630)
    del permute_572
    buf631 = empty((1536, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (1536, 1568), (1, 1536), 0), view_1, out=buf631)
    del view_1
    buf632 = empty((1, 1536), device='cpu', dtype=torch.float32)
    buf633 = buf625; del buf625  # reuse
    buf634 = buf624; del buf624  # reuse
    buf635 = empty((256, ), device='cpu', dtype=torch.float32)
    buf636 = empty((256, ), device='cpu', dtype=torch.float32)
    buf637 = reinterpret_tensor(buf616, (8, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf616  # reuse
    cpp_fused_convolution_backward_native_layer_norm_backward_sum_90(c_void_p(buf637.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf636.data_ptr()))
    del buf629
    del buf630
    del buf633
    del buf634
    del div_61
    del mul
    del primals_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf638 = aten.convolution_backward(buf637, primals_307, primals_1, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf637
    del primals_1
    del primals_307
    buf639 = buf638[1]
    buf640 = buf638[2]
    return (buf639, buf640, buf635, buf636, reinterpret_tensor(buf631, (1536, 256), (256, 1), 0), reinterpret_tensor(buf632, (1536, ), (1, ), 0), buf626, buf627, reinterpret_tensor(buf622, (196, 196), (196, 1), 0), reinterpret_tensor(buf620, (196, ), (1, ), 0), reinterpret_tensor(buf618, (256, 768), (768, 1), 0), reinterpret_tensor(buf619, (256, ), (1, ), 0), buf614, buf615, reinterpret_tensor(buf610, (1536, 256), (256, 1), 0), reinterpret_tensor(buf611, (1536, ), (1, ), 0), buf605, buf606, reinterpret_tensor(buf601, (196, 196), (196, 1), 0), reinterpret_tensor(buf599, (196, ), (1, ), 0), reinterpret_tensor(buf597, (256, 768), (768, 1), 0), reinterpret_tensor(buf598, (256, ), (1, ), 0), buf593, buf594, reinterpret_tensor(buf589, (1536, 256), (256, 1), 0), reinterpret_tensor(buf590, (1536, ), (1, ), 0), buf584, buf585, reinterpret_tensor(buf580, (196, 196), (196, 1), 0), reinterpret_tensor(buf578, (196, ), (1, ), 0), reinterpret_tensor(buf576, (256, 768), (768, 1), 0), reinterpret_tensor(buf577, (256, ), (1, ), 0), buf572, buf573, reinterpret_tensor(buf568, (1536, 256), (256, 1), 0), reinterpret_tensor(buf569, (1536, ), (1, ), 0), buf563, buf564, reinterpret_tensor(buf559, (196, 196), (196, 1), 0), reinterpret_tensor(buf557, (196, ), (1, ), 0), reinterpret_tensor(buf555, (256, 768), (768, 1), 0), reinterpret_tensor(buf556, (256, ), (1, ), 0), buf551, buf552, reinterpret_tensor(buf547, (1536, 256), (256, 1), 0), reinterpret_tensor(buf548, (1536, ), (1, ), 0), buf542, buf543, reinterpret_tensor(buf538, (196, 196), (196, 1), 0), reinterpret_tensor(buf536, (196, ), (1, ), 0), reinterpret_tensor(buf534, (256, 768), (768, 1), 0), reinterpret_tensor(buf535, (256, ), (1, ), 0), buf530, buf531, reinterpret_tensor(buf526, (1536, 256), (256, 1), 0), reinterpret_tensor(buf527, (1536, ), (1, ), 0), buf521, buf522, reinterpret_tensor(buf517, (196, 196), (196, 1), 0), reinterpret_tensor(buf515, (196, ), (1, ), 0), reinterpret_tensor(buf513, (256, 768), (768, 1), 0), reinterpret_tensor(buf514, (256, ), (1, ), 0), buf509, buf510, reinterpret_tensor(buf505, (1536, 256), (256, 1), 0), reinterpret_tensor(buf506, (1536, ), (1, ), 0), buf500, buf501, reinterpret_tensor(buf496, (196, 196), (196, 1), 0), reinterpret_tensor(buf494, (196, ), (1, ), 0), reinterpret_tensor(buf492, (256, 768), (768, 1), 0), reinterpret_tensor(buf493, (256, ), (1, ), 0), buf488, buf489, reinterpret_tensor(buf484, (1536, 256), (256, 1), 0), reinterpret_tensor(buf485, (1536, ), (1, ), 0), buf479, buf480, reinterpret_tensor(buf475, (196, 196), (196, 1), 0), reinterpret_tensor(buf473, (196, ), (1, ), 0), reinterpret_tensor(buf471, (256, 768), (768, 1), 0), reinterpret_tensor(buf472, (256, ), (1, ), 0), buf467, buf468, reinterpret_tensor(buf463, (1536, 256), (256, 1), 0), reinterpret_tensor(buf464, (1536, ), (1, ), 0), buf458, buf459, reinterpret_tensor(buf454, (196, 196), (196, 1), 0), reinterpret_tensor(buf452, (196, ), (1, ), 0), reinterpret_tensor(buf450, (256, 768), (768, 1), 0), reinterpret_tensor(buf451, (256, ), (1, ), 0), buf446, buf447, reinterpret_tensor(buf442, (1536, 256), (256, 1), 0), reinterpret_tensor(buf443, (1536, ), (1, ), 0), buf437, buf438, reinterpret_tensor(buf433, (196, 196), (196, 1), 0), reinterpret_tensor(buf431, (196, ), (1, ), 0), reinterpret_tensor(buf429, (256, 768), (768, 1), 0), reinterpret_tensor(buf430, (256, ), (1, ), 0), buf425, buf426, reinterpret_tensor(buf421, (1536, 256), (256, 1), 0), reinterpret_tensor(buf422, (1536, ), (1, ), 0), buf416, buf417, reinterpret_tensor(buf412, (196, 196), (196, 1), 0), reinterpret_tensor(buf410, (196, ), (1, ), 0), reinterpret_tensor(buf408, (256, 768), (768, 1), 0), reinterpret_tensor(buf409, (256, ), (1, ), 0), buf404, buf405, reinterpret_tensor(buf400, (1536, 256), (256, 1), 0), reinterpret_tensor(buf401, (1536, ), (1, ), 0), buf395, buf396, reinterpret_tensor(buf391, (196, 196), (196, 1), 0), reinterpret_tensor(buf389, (196, ), (1, ), 0), reinterpret_tensor(buf387, (256, 768), (768, 1), 0), reinterpret_tensor(buf388, (256, ), (1, ), 0), buf383, buf384, reinterpret_tensor(buf379, (1536, 256), (256, 1), 0), reinterpret_tensor(buf380, (1536, ), (1, ), 0), buf374, buf375, reinterpret_tensor(buf370, (196, 196), (196, 1), 0), reinterpret_tensor(buf368, (196, ), (1, ), 0), reinterpret_tensor(buf366, (256, 768), (768, 1), 0), reinterpret_tensor(buf367, (256, ), (1, ), 0), buf362, buf363, reinterpret_tensor(buf358, (1536, 256), (256, 1), 0), reinterpret_tensor(buf359, (1536, ), (1, ), 0), buf353, buf354, reinterpret_tensor(buf349, (196, 196), (196, 1), 0), reinterpret_tensor(buf347, (196, ), (1, ), 0), reinterpret_tensor(buf345, (256, 768), (768, 1), 0), reinterpret_tensor(buf346, (256, ), (1, ), 0), buf341, buf342, reinterpret_tensor(buf337, (1536, 256), (256, 1), 0), reinterpret_tensor(buf338, (1536, ), (1, ), 0), buf332, buf333, reinterpret_tensor(buf328, (196, 196), (196, 1), 0), reinterpret_tensor(buf326, (196, ), (1, ), 0), reinterpret_tensor(buf324, (256, 768), (768, 1), 0), reinterpret_tensor(buf325, (256, ), (1, ), 0), buf320, buf321, reinterpret_tensor(buf316, (1536, 256), (256, 1), 0), reinterpret_tensor(buf317, (1536, ), (1, ), 0), buf311, buf312, reinterpret_tensor(buf307, (196, 196), (196, 1), 0), reinterpret_tensor(buf305, (196, ), (1, ), 0), reinterpret_tensor(buf303, (256, 768), (768, 1), 0), reinterpret_tensor(buf304, (256, ), (1, ), 0), buf299, buf300, reinterpret_tensor(buf295, (1536, 256), (256, 1), 0), reinterpret_tensor(buf296, (1536, ), (1, ), 0), buf290, buf291, reinterpret_tensor(buf286, (196, 196), (196, 1), 0), reinterpret_tensor(buf284, (196, ), (1, ), 0), reinterpret_tensor(buf282, (256, 768), (768, 1), 0), reinterpret_tensor(buf283, (256, ), (1, ), 0), buf278, buf279, reinterpret_tensor(buf274, (1536, 256), (256, 1), 0), reinterpret_tensor(buf275, (1536, ), (1, ), 0), buf269, buf270, reinterpret_tensor(buf265, (196, 196), (196, 1), 0), reinterpret_tensor(buf263, (196, ), (1, ), 0), reinterpret_tensor(buf261, (256, 768), (768, 1), 0), reinterpret_tensor(buf262, (256, ), (1, ), 0), buf257, buf258, reinterpret_tensor(buf253, (1536, 256), (256, 1), 0), reinterpret_tensor(buf254, (1536, ), (1, ), 0), buf248, buf249, reinterpret_tensor(buf244, (196, 196), (196, 1), 0), reinterpret_tensor(buf242, (196, ), (1, ), 0), reinterpret_tensor(buf240, (256, 768), (768, 1), 0), reinterpret_tensor(buf241, (256, ), (1, ), 0), buf236, buf237, reinterpret_tensor(buf232, (1536, 256), (256, 1), 0), reinterpret_tensor(buf233, (1536, ), (1, ), 0), buf227, buf228, reinterpret_tensor(buf223, (196, 196), (196, 1), 0), reinterpret_tensor(buf221, (196, ), (1, ), 0), reinterpret_tensor(buf219, (256, 768), (768, 1), 0), reinterpret_tensor(buf220, (256, ), (1, ), 0), buf215, buf216, reinterpret_tensor(buf211, (1536, 256), (256, 1), 0), reinterpret_tensor(buf212, (1536, ), (1, ), 0), buf206, buf207, reinterpret_tensor(buf202, (196, 196), (196, 1), 0), reinterpret_tensor(buf200, (196, ), (1, ), 0), reinterpret_tensor(buf198, (256, 768), (768, 1), 0), reinterpret_tensor(buf199, (256, ), (1, ), 0), buf194, buf195, reinterpret_tensor(buf190, (1536, 256), (256, 1), 0), reinterpret_tensor(buf191, (1536, ), (1, ), 0), buf185, buf186, reinterpret_tensor(buf181, (196, 196), (196, 1), 0), reinterpret_tensor(buf179, (196, ), (1, ), 0), reinterpret_tensor(buf177, (256, 768), (768, 1), 0), reinterpret_tensor(buf178, (256, ), (1, ), 0), buf173, buf174, reinterpret_tensor(buf169, (1536, 256), (256, 1), 0), reinterpret_tensor(buf170, (1536, ), (1, ), 0), buf164, buf165, reinterpret_tensor(buf160, (196, 196), (196, 1), 0), reinterpret_tensor(buf158, (196, ), (1, ), 0), reinterpret_tensor(buf156, (256, 768), (768, 1), 0), reinterpret_tensor(buf157, (256, ), (1, ), 0), buf152, buf153, reinterpret_tensor(buf148, (1536, 256), (256, 1), 0), reinterpret_tensor(buf149, (1536, ), (1, ), 0), buf143, buf144, reinterpret_tensor(buf139, (196, 196), (196, 1), 0), reinterpret_tensor(buf137, (196, ), (1, ), 0), reinterpret_tensor(buf135, (256, 768), (768, 1), 0), reinterpret_tensor(buf136, (256, ), (1, ), 0), buf131, buf132, reinterpret_tensor(buf127, (1536, 256), (256, 1), 0), reinterpret_tensor(buf128, (1536, ), (1, ), 0), buf122, buf123, reinterpret_tensor(buf118, (196, 196), (196, 1), 0), reinterpret_tensor(buf116, (196, ), (1, ), 0), reinterpret_tensor(buf114, (256, 768), (768, 1), 0), reinterpret_tensor(buf115, (256, ), (1, ), 0), buf110, buf111, reinterpret_tensor(buf106, (1536, 256), (256, 1), 0), reinterpret_tensor(buf107, (1536, ), (1, ), 0), buf101, buf102, reinterpret_tensor(buf97, (196, 196), (196, 1), 0), reinterpret_tensor(buf95, (196, ), (1, ), 0), reinterpret_tensor(buf93, (256, 768), (768, 1), 0), reinterpret_tensor(buf94, (256, ), (1, ), 0), buf89, buf90, reinterpret_tensor(buf85, (1536, 256), (256, 1), 0), reinterpret_tensor(buf86, (1536, ), (1, ), 0), buf80, buf81, reinterpret_tensor(buf76, (196, 196), (196, 1), 0), reinterpret_tensor(buf74, (196, ), (1, ), 0), reinterpret_tensor(buf72, (256, 768), (768, 1), 0), reinterpret_tensor(buf73, (256, ), (1, ), 0), buf68, buf69, reinterpret_tensor(buf64, (1536, 256), (256, 1), 0), reinterpret_tensor(buf65, (1536, ), (1, ), 0), buf59, buf60, reinterpret_tensor(buf55, (196, 196), (196, 1), 0), reinterpret_tensor(buf53, (196, ), (1, ), 0), reinterpret_tensor(buf51, (256, 768), (768, 1), 0), reinterpret_tensor(buf52, (256, ), (1, ), 0), buf47, buf48, reinterpret_tensor(buf43, (1536, 256), (256, 1), 0), reinterpret_tensor(buf44, (1536, ), (1, ), 0), buf38, buf39, reinterpret_tensor(buf34, (196, 196), (196, 1), 0), reinterpret_tensor(buf32, (196, ), (1, ), 0), reinterpret_tensor(buf30, (256, 768), (768, 1), 0), reinterpret_tensor(buf31, (256, ), (1, ), 0), buf26, buf27, reinterpret_tensor(buf22, (1536, 256), (256, 1), 0), reinterpret_tensor(buf23, (1536, ), (1, ), 0), buf17, buf18, reinterpret_tensor(buf13, (196, 196), (196, 1), 0), reinterpret_tensor(buf11, (196, ), (1, ), 0), reinterpret_tensor(buf9, (256, 768), (768, 1), 0), reinterpret_tensor(buf10, (256, ), (1, ), 0), buf6, buf7, reinterpret_tensor(buf1, (1000, 256), (256, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((256, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((196, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_5 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_3 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_8 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_8 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_13 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_1 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_14 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_21 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_2 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_24 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_20 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_29 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_3 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_32 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_8 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_26 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_37 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_4 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_32 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_45 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_33 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_5 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_48 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_12 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_38 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_53 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_6 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_43 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_44 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_61 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_7 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_64 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_50 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_69 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_8 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_56 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_77 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_9 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_20 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_62 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_85 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_63 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_10 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_68 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_93 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_11 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_96 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_24 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_74 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_101 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_12 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_104 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_79 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_80 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_109 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_13 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_112 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_86 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_117 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_87 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_14 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_92 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_125 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_15 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_128 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_97 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_32 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_98 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_133 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_99 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_16 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_136 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_103 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_104 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_141 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_105 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_17 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_144 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_109 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_36 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_110 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_149 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_111 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_18 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_113 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_152 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_38 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_116 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_157 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_19 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_160 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_121 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_122 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_165 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_123 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_20 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_168 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_127 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_128 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_173 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_129 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_21 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_131 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_176 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_133 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_44 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_134 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_181 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_135 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_22 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_137 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_184 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_139 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_140 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_189 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_141 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_23 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_143 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_145 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_48 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_146 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_197 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_147 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_24 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_149 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_200 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_151 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_50 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_152 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_205 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_153 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_25 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_155 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_208 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_157 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_158 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_213 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_159 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_26 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_161 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_216 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_163 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_54 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_164 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_221 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_165 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_27 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_167 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_224 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_169 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_56 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_170 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_229 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_171 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_28 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_173 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_232 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_175 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((1568, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    getitem_176 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cpu', dtype=torch.float32)
    mul_237 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_177 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    mm_29 = rand_strided((6144, 196), (196, 1), device='cpu', dtype=torch.float32)
    view_179 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_240 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    clone_151 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_152 = rand_strided((1000, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_156 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_163 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_177 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_180 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_198 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_205 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_247 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_264 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_268 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_275 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_282 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_292 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_296 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_303 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_306 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_317 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_320 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_324 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_331 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_334 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_338 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_345 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_352 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_359 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_362 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_366 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_373 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_380 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_387 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_390 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_394 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_401 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_404 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_38 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_418 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_422 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_429 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_432 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_436 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_443 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_446 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_450 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_457 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_44 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_460 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_474 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_478 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_485 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_488 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_492 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_499 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_50 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_502 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_506 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_513 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_516 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_53 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_520 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_527 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_530 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_534 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_541 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_56 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_544 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_548 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_555 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_558 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_59 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_562 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_569 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_572 = rand_strided((1536, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_7, primals_10, primals_13, primals_17, primals_20, primals_23, primals_27, primals_30, primals_33, primals_37, primals_40, primals_43, primals_47, primals_50, primals_53, primals_57, primals_60, primals_63, primals_67, primals_70, primals_73, primals_77, primals_80, primals_83, primals_87, primals_90, primals_93, primals_97, primals_100, primals_103, primals_107, primals_110, primals_113, primals_117, primals_120, primals_123, primals_127, primals_130, primals_133, primals_137, primals_140, primals_143, primals_147, primals_150, primals_153, primals_157, primals_160, primals_163, primals_167, primals_170, primals_173, primals_177, primals_180, primals_183, primals_187, primals_190, primals_193, primals_197, primals_200, primals_203, primals_207, primals_210, primals_213, primals_217, primals_220, primals_223, primals_227, primals_230, primals_233, primals_237, primals_240, primals_243, primals_247, primals_250, primals_253, primals_257, primals_260, primals_263, primals_267, primals_270, primals_273, primals_277, primals_280, primals_283, primals_287, primals_290, primals_293, primals_297, primals_300, primals_303, primals_307, mul, view_1, addmm, getitem_2, mul_5, view_3, mm, view_5, mul_8, view_7, addmm_2, getitem_8, mul_13, view_9, mm_1, view_11, mul_16, view_13, addmm_4, getitem_14, mul_21, view_15, mm_2, view_17, mul_24, view_19, addmm_6, getitem_20, mul_29, view_21, mm_3, view_23, mul_32, view_25, addmm_8, getitem_26, mul_37, view_27, mm_4, view_29, mul_40, view_31, addmm_10, getitem_32, mul_45, view_33, mm_5, view_35, mul_48, view_37, addmm_12, getitem_38, mul_53, view_39, mm_6, view_41, mul_56, view_43, addmm_14, getitem_44, mul_61, view_45, mm_7, view_47, mul_64, view_49, addmm_16, getitem_50, mul_69, view_51, mm_8, view_53, mul_72, view_55, addmm_18, getitem_56, mul_77, view_57, mm_9, view_59, mul_80, view_61, addmm_20, getitem_62, mul_85, view_63, mm_10, view_65, mul_88, view_67, addmm_22, getitem_68, mul_93, view_69, mm_11, view_71, mul_96, view_73, addmm_24, getitem_74, mul_101, view_75, mm_12, view_77, mul_104, view_79, addmm_26, getitem_80, mul_109, view_81, mm_13, view_83, mul_112, view_85, addmm_28, getitem_86, mul_117, view_87, mm_14, view_89, mul_120, view_91, addmm_30, getitem_92, mul_125, view_93, mm_15, view_95, mul_128, view_97, addmm_32, getitem_98, mul_133, view_99, mm_16, view_101, mul_136, view_103, addmm_34, getitem_104, mul_141, view_105, mm_17, view_107, mul_144, view_109, addmm_36, getitem_110, mul_149, view_111, mm_18, view_113, mul_152, view_115, addmm_38, getitem_116, mul_157, view_117, mm_19, view_119, mul_160, view_121, addmm_40, getitem_122, mul_165, view_123, mm_20, view_125, mul_168, view_127, addmm_42, getitem_128, mul_173, view_129, mm_21, view_131, mul_176, view_133, addmm_44, getitem_134, mul_181, view_135, mm_22, view_137, mul_184, view_139, addmm_46, getitem_140, mul_189, view_141, mm_23, view_143, mul_192, view_145, addmm_48, getitem_146, mul_197, view_147, mm_24, view_149, mul_200, view_151, addmm_50, getitem_152, mul_205, view_153, mm_25, view_155, mul_208, view_157, addmm_52, getitem_158, mul_213, view_159, mm_26, view_161, mul_216, view_163, addmm_54, getitem_164, mul_221, view_165, mm_27, view_167, mul_224, view_169, addmm_56, getitem_170, mul_229, view_171, mm_28, view_173, mul_232, view_175, addmm_58, getitem_176, mul_237, view_177, mm_29, view_179, mul_240, clone_151, permute_152, div_1, permute_156, permute_163, div_2, permute_166, div_3, permute_170, permute_177, div_4, permute_180, div_5, permute_184, permute_191, div_6, permute_194, div_7, permute_198, permute_205, div_8, permute_208, div_9, permute_212, permute_219, div_10, permute_222, div_11, permute_226, permute_233, div_12, permute_236, div_13, permute_240, permute_247, div_14, permute_250, div_15, permute_254, permute_261, div_16, permute_264, div_17, permute_268, permute_275, div_18, permute_278, div_19, permute_282, permute_289, div_20, permute_292, div_21, permute_296, permute_303, div_22, permute_306, div_23, permute_310, permute_317, div_24, permute_320, div_25, permute_324, permute_331, div_26, permute_334, div_27, permute_338, permute_345, div_28, permute_348, div_29, permute_352, permute_359, div_30, permute_362, div_31, permute_366, permute_373, div_32, permute_376, div_33, permute_380, permute_387, div_34, permute_390, div_35, permute_394, permute_401, div_36, permute_404, div_37, permute_408, permute_415, div_38, permute_418, div_39, permute_422, permute_429, div_40, permute_432, div_41, permute_436, permute_443, div_42, permute_446, div_43, permute_450, permute_457, div_44, permute_460, div_45, permute_464, permute_471, div_46, permute_474, div_47, permute_478, permute_485, div_48, permute_488, div_49, permute_492, permute_499, div_50, permute_502, div_51, permute_506, permute_513, div_52, permute_516, div_53, permute_520, permute_527, div_54, permute_530, div_55, permute_534, permute_541, div_56, permute_544, div_57, permute_548, permute_555, div_58, permute_558, div_59, permute_562, permute_569, div_60, permute_572, div_61, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmlp_s16_224', benchmark_compiled_module)
