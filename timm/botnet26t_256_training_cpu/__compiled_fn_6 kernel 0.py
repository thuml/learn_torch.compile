
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


cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(0.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                            auto tmp10 = tmp8 - tmp9;
                            auto tmp11 = tmp7 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp7;
                            tmp_acc1_vec = tmp_acc1_vec + tmp11;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.001953125);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp16 = tmp15 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp10 * tmp17;
                        auto tmp19 = tmp7 - tmp18;
                        auto tmp21 = tmp20 * tmp13;
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = tmp15 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        tmp25.store(out_ptr4 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x3) + (4096L*x2) + (32768L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (4096L*x2) + (32768L*x1)));
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((128L*x3) + (1024L*x2) + (8192L*(c10::div_floor_integer((x3 + (8L*x2) + (64L*x0) + (64L*x0_inner)), 8192L))) + (32768L*x1) + (static_cast<long>(c10::div_floor_integer((x3 + (8L*x2) + (64L*x0) + (64L*x0_inner)), 64L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                        auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((128L*x1) + (8192L*(c10::div_floor_integer((x1 + (64L*x2) + (64L*x2_inner)), 8192L))) + (32768L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.001953125);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp13 * tmp13;
                        auto tmp15 = tmp12 * tmp14;
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp5 - tmp16;
                        auto tmp19 = tmp18 * tmp11;
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = tmp13 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp23.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr3[static_cast<long>(x1 + (64L*x2) + (64L*x2_inner) + (32768L*x0))] = tmpbuf[x2_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (8L*x3) + (64L*x2) + (512L*x1) + (4096L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (8L*x3) + (64L*x2) + (512L*x1) + (4096L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (8L*x1) + (64L*x2) + (512L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(15L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(16);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))));
                            auto tmp5 = static_cast<long>(135);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L));
                                auto tmp9 = static_cast<long>(8);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L));
                                    auto tmp13 = static_cast<long>(7);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr1[static_cast<long>((-7L) + (8L*(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L))) + (64L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L)))];
                                        return tmp16;
                                    }
                                    ;
                                    auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                    auto tmp18 = static_cast<float>(0.0);
                                    auto tmp19 = tmp14 ? tmp17 : tmp18;
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp21 = static_cast<float>(0.0);
                                auto tmp22 = tmp10 ? tmp20 : tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr2[static_cast<long>(x1 + (15L*x0))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (8L*x2) + (64L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (8L*x2) + (64L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(15L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(16);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))));
                            auto tmp5 = static_cast<long>(135);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L));
                                auto tmp9 = static_cast<long>(8);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L));
                                    auto tmp13 = static_cast<long>(7);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr0[static_cast<long>((-7L) + (8L*(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L))) + (64L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L)))];
                                        return tmp16;
                                    }
                                    ;
                                    auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                    auto tmp18 = static_cast<float>(0.0);
                                    auto tmp19 = tmp14 ? tmp17 : tmp18;
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp21 = static_cast<float>(0.0);
                                auto tmp22 = tmp10 ? tmp20 : tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x1 + (15L*x0))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.08838834764831845);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1536L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(512);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (1024L*x2) + (1024L*x2_inner) + (8192L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((128L*x2) + (128L*x2_inner) + (1024L*x1) + (8192L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((128L*x2) + (128L*x2_inner) + (1024L*x1) + (8192L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(1024);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-32768L) + x2 + (8L*x1) + (64L*x3) + (32768L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(1536);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((128L*x2) + (128L*x2_inner) + (1024L*x1) + (8192L*(static_cast<long>(c10::div_floor_integer(((-65536L) + x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (8L*x1) + (64L*x3) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp4 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = static_cast<float>(64.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 / tmp7;
                            auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp4);
                            auto tmp11 = tmp9 + tmp10;
                            auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                            auto tmp15 = tmp13 - tmp14;
                            auto tmp16 = tmp12 * tmp15;
                            auto tmp19 = tmp17 - tmp18;
                            auto tmp20 = tmp12 * tmp19;
                            tmp_acc0_vec = tmp_acc0_vec + tmp12;
                            tmp_acc1_vec = tmp_acc1_vec + tmp16;
                            tmp_acc2_vec = tmp_acc2_vec + tmp20;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp4 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = static_cast<float>(64.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 / tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp4);
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp17 = static_cast<float>(0.001953125);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp23 = tmp15 * tmp22;
                        auto tmp24 = tmp12 - tmp23;
                        auto tmp26 = tmp25 * tmp18;
                        auto tmp27 = tmp24 - tmp26;
                        auto tmp30 = tmp28 - tmp29;
                        auto tmp32 = tmp31 * tmp18;
                        auto tmp34 = tmp33 * tmp33;
                        auto tmp35 = tmp32 * tmp34;
                        auto tmp36 = tmp30 * tmp35;
                        auto tmp37 = tmp12 - tmp36;
                        auto tmp38 = tmp37 - tmp26;
                        tmp27.store(out_ptr3 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        tmp38.store(out_ptr4 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
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
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x3, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x3, 2L)))))) >= 0L) ? 0L : 8L)) + (4096L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (4096L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 8L)) + (32768L*x0))];
                            auto tmp1 = tmp0 / 4;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(8L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(8L, 1L + (c10::div_floor_integer(x3, 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            out_ptr3[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (131072L*x0))] = tmp10;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(16L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (16L*x1) + (256L*x2) + (4096L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr1[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp14 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = tmp10 ? tmp20 : tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    out_ptr2[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr0[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp14 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = tmp10 ? tmp20 : tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    out_ptr1[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.08838834764831845);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_13 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1536L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(512);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (2048L*x2) + (2048L*x2_inner) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((128L*x2) + (128L*x2_inner) + (2048L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((128L*x2) + (128L*x2_inner) + (2048L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(1024);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-131072L) + x2 + (16L*x1) + (256L*x3) + (131072L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(1536);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((128L*x2) + (128L*x2_inner) + (2048L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(((-262144L) + x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.00048828125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x3) + (4096L*x2) + (65536L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x3) + (4096L*x2) + (65536L*x1)));
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((64L*x3) + (1024L*x2) + (16384L*(c10::div_floor_integer((x3 + (16L*x2) + (256L*x0) + (256L*x0_inner)), 16384L))) + (65536L*x1) + (static_cast<long>(c10::div_floor_integer((x3 + (16L*x2) + (256L*x0) + (256L*x0_inner)), 256L)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((64L*x1) + (16384L*(c10::div_floor_integer((x1 + (256L*x2) + (256L*x2_inner)), 16384L))) + (65536L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.00048828125);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp13 * tmp13;
                        auto tmp15 = tmp12 * tmp14;
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp5 - tmp16;
                        auto tmp19 = tmp18 * tmp11;
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = tmp13 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp23.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr3[static_cast<long>(x1 + (256L*x2) + (256L*x2_inner) + (65536L*x0))] = tmpbuf[x2_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(16L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (16L*x1) + (256L*x2) + (4096L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr1[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp14 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = tmp10 ? tmp20 : tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    out_ptr2[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr0[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp14 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = tmp10 ? tmp20 : tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    out_ptr1[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_20 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(256);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (1024L*x2) + (1024L*x2_inner) + (16384L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((64L*x2) + (64L*x2_inner) + (1024L*x1) + (16384L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((64L*x2) + (64L*x2_inner) + (1024L*x1) + (16384L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(512);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-65536L) + x2 + (16L*x1) + (256L*x3) + (65536L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(768);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((64L*x2) + (64L*x2_inner) + (1024L*x1) + (16384L*(static_cast<long>(c10::div_floor_integer(((-131072L) + x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00048828125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.0001220703125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0001220703125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(3.0517578125e-05);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(3.0517578125e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(7.62939453125e-06);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(7.62939453125e-06);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(7.62939453125e-06);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_195, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, view_7, view_13, bmm_1, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, view_31, view_37, view_47, avg_pool2d, squeeze_76, relu_22, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_23, convolution_28, squeeze_85, relu_24, view_55, view_61, bmm_5, squeeze_88, relu_25, convolution_30, squeeze_91, clone_21, permute_25, le, unsqueeze_126, unsqueeze_138, permute_30, permute_31, alias_36, permute_35, permute_41, permute_43, permute_44, unsqueeze_150, unsqueeze_162, unsqueeze_174, unsqueeze_186, permute_48, permute_49, alias_46, permute_53, permute_59, permute_61, permute_62, unsqueeze_198, unsqueeze_210, unsqueeze_222, permute_66, permute_67, alias_56, permute_71, permute_77, permute_79, permute_80, unsqueeze_234, unsqueeze_246, unsqueeze_258, unsqueeze_270, unsqueeze_282, unsqueeze_294, unsqueeze_306, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (24, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_57, (2048, ), (1, ))
    assert_size_stride(primals_59, (2048, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_69, (24, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_70, (32, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(primals_71, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_72, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_74, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_77, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_78, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_81, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_82, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_84, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_85, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_87, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_88, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_90, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_91, (768, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_93, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_94, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_95, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_96, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_97, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_98, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_99, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_195, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(squeeze_1, (24, ), (1, ))
    assert_size_stride(relu, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(convolution_1, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(relu_1, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_2, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(getitem_6, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(getitem_7, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_3, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(relu_3, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_4, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(relu_4, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_5, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_6, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(relu_5, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_7, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(relu_6, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_8, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_7, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_9, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_28, (256, ), (1, ))
    assert_size_stride(relu_8, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_10, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(relu_9, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_11, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(relu_10, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_12, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_37, (512, ), (1, ))
    assert_size_stride(convolution_13, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(relu_11, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_14, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(relu_12, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_15, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(relu_13, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_16, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_49, (512, ), (1, ))
    assert_size_stride(relu_14, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_17, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(relu_15, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_18, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_55, (256, ), (1, ))
    assert_size_stride(relu_16, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_19, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_58, (1024, ), (1, ))
    assert_size_stride(convolution_20, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_61, (1024, ), (1, ))
    assert_size_stride(relu_17, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_21, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_64, (256, ), (1, ))
    assert_size_stride(relu_18, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(view_7, (8192, 64), (64, 1))
    assert_size_stride(view_13, (8192, 64), (64, 1))
    assert_size_stride(bmm_1, (32, 256, 64), (16384, 64, 1))
    assert_size_stride(squeeze_67, (256, ), (1, ))
    assert_size_stride(relu_19, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_23, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_70, (1024, ), (1, ))
    assert_size_stride(relu_20, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_24, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_73, (512, ), (1, ))
    assert_size_stride(relu_21, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(view_31, (8192, 128), (128, 1))
    assert_size_stride(view_37, (8192, 128), (128, 1))
    assert_size_stride(view_47, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(avg_pool2d, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(relu_22, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_26, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_79, (2048, ), (1, ))
    assert_size_stride(convolution_27, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_82, (2048, ), (1, ))
    assert_size_stride(relu_23, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_28, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_85, (512, ), (1, ))
    assert_size_stride(relu_24, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(view_55, (2048, 128), (128, 1))
    assert_size_stride(view_61, (2048, 128), (128, 1))
    assert_size_stride(bmm_5, (32, 64, 128), (8192, 128, 1))
    assert_size_stride(squeeze_88, (512, ), (1, ))
    assert_size_stride(relu_25, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_30, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_91, (2048, ), (1, ))
    assert_size_stride(clone_21, (8, 2048), (2048, 1))
    assert_size_stride(permute_25, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(unsqueeze_126, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_138, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_30, (32, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_31, (32, 128, 64), (8192, 64, 1))
    assert_size_stride(alias_36, (32, 64, 64), (4096, 64, 1))
    assert_size_stride(permute_35, (15, 128), (128, 1))
    assert_size_stride(permute_41, (15, 128), (128, 1))
    assert_size_stride(permute_43, (32, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_44, (32, 64, 128), (8192, 1, 64))
    assert_size_stride(unsqueeze_150, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_162, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_174, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_186, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_48, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_49, (32, 128, 256), (32768, 256, 1))
    assert_size_stride(alias_46, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_53, (31, 128), (128, 1))
    assert_size_stride(permute_59, (31, 128), (128, 1))
    assert_size_stride(permute_61, (32, 128, 256), (32768, 256, 1))
    assert_size_stride(permute_62, (32, 256, 128), (32768, 1, 256))
    assert_size_stride(unsqueeze_198, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_210, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(permute_66, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_67, (32, 64, 256), (16384, 256, 1))
    assert_size_stride(alias_56, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_71, (31, 64), (64, 1))
    assert_size_stride(permute_77, (31, 64), (64, 1))
    assert_size_stride(permute_79, (32, 64, 256), (16384, 256, 1))
    assert_size_stride(permute_80, (32, 256, 64), (16384, 1, 256))
    assert_size_stride(unsqueeze_234, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_246, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_270, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_342, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_486, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_25, out=buf0)
    del permute_25
    buf1 = empty((1000, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_21, out=buf1)
    del clone_21
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf5 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_126.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del convolution_30
    del primals_67
    del squeeze_91
    del tangents_1
    del unsqueeze_126
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, relu_25, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_99
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((512, ), device='cpu', dtype=torch.float32)
    buf11 = empty((512, ), device='cpu', dtype=torch.float32)
    buf12 = empty((512, ), device='cpu', dtype=torch.float32)
    buf13 = empty((8, 512, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_threshold_backward_1(c_void_p(relu_25.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(bmm_5.data_ptr()), c_void_p(unsqueeze_138.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del bmm_5
    del primals_65
    del relu_25
    del squeeze_88
    del unsqueeze_138
    buf14 = reinterpret_tensor(buf8, (32, 64, 128), (8192, 128, 1), 0); del buf8  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_30, reinterpret_tensor(buf13, (32, 64, 128), (8192, 1, 64), 0), out=buf14)
    del permute_30
    buf15 = empty((32, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (32, 64, 128), (8192, 1, 64), 0), permute_31, out=buf15)
    del permute_31
    buf16 = reinterpret_tensor(buf4, (32, 64, 1), (64, 1, 2048), 0); del buf4  # reuse
    buf17 = empty((32, 8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf18 = empty((2048, 15), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_2(c_void_p(buf15.data_ptr()), c_void_p(alias_36.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = empty((15, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (15, 2048), (1, 15), 0), view_61, out=buf19)
    del view_61
    buf20 = reinterpret_tensor(buf13, (2048, 128), (128, 1), 0); del buf13  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf18, permute_35, out=buf20)
    del permute_35
    buf21 = buf17; del buf17  # reuse
    buf22 = buf18; del buf18  # reuse
    cpp_fused_sum_view_3(c_void_p(buf15.data_ptr()), c_void_p(alias_36.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del buf21
    buf23 = empty((15, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (15, 2048), (1, 15), 0), view_55, out=buf23)
    del view_55
    buf24 = empty((2048, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf22, permute_41, out=buf24)
    del buf22
    del permute_41
    buf25 = buf15; del buf15  # reuse
    cpp_fused__softmax_backward_data_mul_4(c_void_p(buf25.data_ptr()), c_void_p(alias_36.data_ptr()), c_void_p(buf16.data_ptr()))
    del alias_36
    buf26 = empty((32, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_43, buf25, out=buf26)
    del permute_43
    buf27 = empty((32, 64, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf25, permute_44, out=buf27)
    del permute_44
    buf28 = empty((8, 1536, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_5(c_void_p(buf20.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf28.data_ptr()))
    del buf14
    del buf20
    del buf24
    del buf26
    del buf27
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf29 = aten.convolution_backward(buf28, relu_24, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf28
    del primals_98
    buf30 = buf29[0]
    buf31 = buf29[1]
    del buf29
    buf32 = buf11; del buf11  # reuse
    buf33 = empty((512, ), device='cpu', dtype=torch.float32)
    buf34 = empty((512, ), device='cpu', dtype=torch.float32)
    buf35 = buf30; del buf30  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf35.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_150.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del convolution_28
    del primals_61
    del relu_24
    del squeeze_85
    del unsqueeze_150
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf36 = aten.convolution_backward(buf35, relu_23, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf35
    del primals_97
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = reinterpret_tensor(buf16, (2048, ), (1, ), 0); del buf16  # reuse
    buf40 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf47 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf41 = buf6; del buf6  # reuse
    buf48 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    buf42 = buf40; del buf40  # reuse
    buf43 = buf41; del buf41  # reuse
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_7(c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_162.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_174.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del buf0
    del convolution_26
    del convolution_27
    del le
    del primals_59
    del relu_23
    del squeeze_82
    del unsqueeze_162
    del unsqueeze_174
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf44 = aten.convolution_backward(buf43, relu_20, primals_96, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_96
    buf45 = buf44[0]
    buf46 = buf44[1]
    del buf44
    buf49 = buf47; del buf47  # reuse
    buf50 = buf48; del buf48  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_8(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_57.data_ptr()))
    del primals_57
    del squeeze_79
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf51 = aten.convolution_backward(buf50, relu_22, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_95
    buf52 = buf51[0]
    buf53 = buf51[1]
    del buf51
    buf54 = buf33; del buf33  # reuse
    buf55 = empty((512, ), device='cpu', dtype=torch.float32)
    buf56 = empty((512, ), device='cpu', dtype=torch.float32)
    buf57 = buf52; del buf52  # reuse
    buf58 = reinterpret_tensor(buf50, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf50  # reuse
    cpp_fused_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf57.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(avg_pool2d.data_ptr()), c_void_p(unsqueeze_186.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()))
    del avg_pool2d
    del buf57
    del primals_55
    del relu_22
    del squeeze_76
    del unsqueeze_186
    buf59 = reinterpret_tensor(buf43, (32, 256, 128), (32768, 128, 1), 0); del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_48, reinterpret_tensor(buf58, (32, 256, 128), (32768, 1, 256), 0), out=buf59)
    del permute_48
    buf60 = empty((32, 256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (32, 256, 128), (32768, 1, 256), 0), permute_49, out=buf60)
    del permute_49
    buf61 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf25, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1), 0); del buf25  # reuse
    buf63 = empty((8192, 31), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_10(c_void_p(buf60.data_ptr()), c_void_p(alias_46.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    buf64 = empty((31, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (31, 8192), (1, 31), 0), view_37, out=buf64)
    del view_37
    buf65 = reinterpret_tensor(buf58, (8192, 128), (128, 1), 0); del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf63, permute_53, out=buf65)
    del permute_53
    buf66 = buf62; del buf62  # reuse
    buf67 = buf63; del buf63  # reuse
    cpp_fused_sum_view_11(c_void_p(buf60.data_ptr()), c_void_p(alias_46.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = empty((31, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (31, 8192), (1, 31), 0), view_31, out=buf68)
    del view_31
    buf69 = reinterpret_tensor(buf37, (8192, 128), (128, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf67, permute_59, out=buf69)
    del permute_59
    buf70 = buf60; del buf60  # reuse
    cpp_fused__softmax_backward_data_mul_12(c_void_p(buf70.data_ptr()), c_void_p(alias_46.data_ptr()), c_void_p(buf61.data_ptr()))
    del alias_46
    buf71 = empty((32, 128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_61, buf70, out=buf71)
    del permute_61
    buf72 = empty((32, 256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf70, permute_62, out=buf72)
    del permute_62
    buf73 = empty((8, 1536, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_13(c_void_p(buf65.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf73.data_ptr()))
    del buf59
    del buf65
    del buf69
    del buf71
    del buf72
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf74 = aten.convolution_backward(buf73, relu_21, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf73
    del primals_94
    buf75 = buf74[0]
    buf76 = buf74[1]
    del buf74
    buf77 = buf55; del buf55  # reuse
    buf78 = empty((512, ), device='cpu', dtype=torch.float32)
    buf79 = empty((512, ), device='cpu', dtype=torch.float32)
    buf80 = buf75; del buf75  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf80.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_198.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del convolution_24
    del primals_51
    del relu_21
    del squeeze_73
    del unsqueeze_198
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf81 = aten.convolution_backward(buf80, relu_20, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf80
    del primals_93
    buf82 = buf81[0]
    buf83 = buf81[1]
    del buf81
    buf84 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf85 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf70, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf70  # reuse
    buf87 = buf85; del buf85  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_15(c_void_p(buf87.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()))
    del convolution_23
    del primals_49
    del squeeze_70
    del unsqueeze_210
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf88 = aten.convolution_backward(buf86, relu_19, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_92
    buf89 = buf88[0]
    buf90 = buf88[1]
    del buf88
    buf91 = empty((256, ), device='cpu', dtype=torch.float32)
    buf92 = empty((256, ), device='cpu', dtype=torch.float32)
    buf93 = empty((256, ), device='cpu', dtype=torch.float32)
    buf94 = empty((8, 256, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_threshold_backward_16(c_void_p(relu_19.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(bmm_1.data_ptr()), c_void_p(unsqueeze_222.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del bmm_1
    del primals_47
    del relu_19
    del squeeze_67
    del unsqueeze_222
    buf95 = reinterpret_tensor(buf89, (32, 256, 64), (16384, 64, 1), 0); del buf89  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_66, reinterpret_tensor(buf94, (32, 256, 64), (16384, 1, 256), 0), out=buf95)
    del permute_66
    buf96 = reinterpret_tensor(buf86, (32, 256, 256), (65536, 256, 1), 0); del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf94, (32, 256, 64), (16384, 1, 256), 0), permute_67, out=buf96)
    del permute_67
    buf97 = buf61; del buf61  # reuse
    buf98 = buf66; del buf66  # reuse
    buf99 = buf67; del buf67  # reuse
    cpp_fused__softmax_backward_data_sum_view_17(c_void_p(buf96.data_ptr()), c_void_p(alias_56.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = empty((31, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (31, 8192), (1, 31), 0), view_13, out=buf100)
    del view_13
    buf101 = reinterpret_tensor(buf94, (8192, 64), (64, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf99, permute_71, out=buf101)
    del permute_71
    buf102 = buf98; del buf98  # reuse
    buf103 = buf99; del buf99  # reuse
    cpp_fused_sum_view_18(c_void_p(buf96.data_ptr()), c_void_p(alias_56.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del buf102
    buf104 = empty((31, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (31, 8192), (1, 31), 0), view_7, out=buf104)
    del view_7
    buf105 = empty((8192, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf103, permute_77, out=buf105)
    del buf103
    del permute_77
    buf106 = buf96; del buf96  # reuse
    cpp_fused__softmax_backward_data_mul_19(c_void_p(buf106.data_ptr()), c_void_p(alias_56.data_ptr()), c_void_p(buf97.data_ptr()))
    del alias_56
    del buf97
    buf107 = empty((32, 64, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_79, buf106, out=buf107)
    del permute_79
    buf108 = empty((32, 256, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf106, permute_80, out=buf108)
    del permute_80
    buf109 = empty((8, 768, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_20(c_void_p(buf101.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf109.data_ptr()))
    del buf101
    del buf105
    del buf107
    del buf108
    del buf95
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf110 = aten.convolution_backward(buf109, relu_18, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf109
    del primals_91
    buf111 = buf110[0]
    buf112 = buf110[1]
    del buf110
    buf113 = buf92; del buf92  # reuse
    buf114 = empty((256, ), device='cpu', dtype=torch.float32)
    buf115 = empty((256, ), device='cpu', dtype=torch.float32)
    buf116 = buf111; del buf111  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf116.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del convolution_21
    del primals_43
    del relu_18
    del squeeze_64
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf117 = aten.convolution_backward(buf116, relu_17, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf116
    del primals_90
    buf118 = buf117[0]
    buf119 = buf117[1]
    del buf117
    buf120 = buf118; del buf118  # reuse
    buf121 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf122 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf128 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf123 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf124 = reinterpret_tensor(buf106, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf106  # reuse
    buf130 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf120.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf130.data_ptr()))
    del buf120
    del buf122
    del buf45
    del buf82
    del convolution_19
    del convolution_20
    del primals_39
    del primals_41
    del relu_17
    del relu_20
    del squeeze_61
    del unsqueeze_246
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf125 = aten.convolution_backward(buf124, relu_14, primals_89, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf124
    del primals_89
    buf126 = buf125[0]
    buf127 = buf125[1]
    del buf125
    buf129 = buf128; del buf128  # reuse
    cpp_fused_native_batch_norm_backward_23(c_void_p(buf129.data_ptr()), c_void_p(squeeze_58.data_ptr()))
    del squeeze_58
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf131 = aten.convolution_backward(buf130, relu_16, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf130
    del primals_88
    buf132 = buf131[0]
    buf133 = buf131[1]
    del buf131
    buf134 = buf114; del buf114  # reuse
    buf135 = empty((256, ), device='cpu', dtype=torch.float32)
    buf136 = empty((256, ), device='cpu', dtype=torch.float32)
    buf137 = buf132; del buf132  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf137.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    del convolution_18
    del primals_37
    del relu_16
    del squeeze_55
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf138 = aten.convolution_backward(buf137, relu_15, primals_87, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf137
    del primals_87
    buf139 = buf138[0]
    buf140 = buf138[1]
    del buf138
    buf141 = buf135; del buf135  # reuse
    buf142 = empty((256, ), device='cpu', dtype=torch.float32)
    buf143 = empty((256, ), device='cpu', dtype=torch.float32)
    buf144 = buf139; del buf139  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf144.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    del convolution_17
    del primals_35
    del relu_15
    del squeeze_52
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf145 = aten.convolution_backward(buf144, relu_14, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf144
    del primals_86
    buf146 = buf145[0]
    buf147 = buf145[1]
    del buf145
    buf148 = buf78; del buf78  # reuse
    buf149 = empty((512, ), device='cpu', dtype=torch.float32)
    buf150 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    buf151 = buf149; del buf149  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_26(c_void_p(buf151.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()))
    del convolution_16
    del primals_33
    del squeeze_49
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf152 = aten.convolution_backward(buf150, relu_13, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_85
    buf153 = buf152[0]
    buf154 = buf152[1]
    del buf152
    buf155 = empty((128, ), device='cpu', dtype=torch.float32)
    buf156 = empty((128, ), device='cpu', dtype=torch.float32)
    buf157 = empty((128, ), device='cpu', dtype=torch.float32)
    buf158 = buf153; del buf153  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf158.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    del convolution_15
    del primals_31
    del relu_13
    del squeeze_46
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf159 = aten.convolution_backward(buf158, relu_12, primals_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf158
    del primals_84
    buf160 = buf159[0]
    buf161 = buf159[1]
    del buf159
    buf162 = buf156; del buf156  # reuse
    buf163 = empty((128, ), device='cpu', dtype=torch.float32)
    buf164 = empty((128, ), device='cpu', dtype=torch.float32)
    buf165 = buf160; del buf160  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf165.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    del convolution_14
    del primals_29
    del relu_12
    del squeeze_43
    del unsqueeze_318
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf166 = aten.convolution_backward(buf165, relu_11, primals_83, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf165
    del primals_83
    buf167 = buf166[0]
    buf168 = buf166[1]
    del buf166
    buf169 = buf126; del buf126  # reuse
    buf170 = empty((512, ), device='cpu', dtype=torch.float32)
    buf171 = empty((512, ), device='cpu', dtype=torch.float32)
    buf177 = empty((512, ), device='cpu', dtype=torch.float32)
    buf172 = empty((512, ), device='cpu', dtype=torch.float32)
    buf173 = buf150; del buf150  # reuse
    buf179 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf169.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf179.data_ptr()))
    del buf146
    del buf167
    del buf169
    del buf171
    del convolution_12
    del convolution_13
    del primals_25
    del primals_27
    del relu_11
    del relu_14
    del squeeze_40
    del unsqueeze_330
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf174 = aten.convolution_backward(buf173, relu_8, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf173
    del primals_82
    buf175 = buf174[0]
    buf176 = buf174[1]
    del buf174
    buf178 = buf177; del buf177  # reuse
    cpp_fused_native_batch_norm_backward_30(c_void_p(buf178.data_ptr()), c_void_p(squeeze_37.data_ptr()))
    del squeeze_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf180 = aten.convolution_backward(buf179, relu_10, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf179
    del primals_81
    buf181 = buf180[0]
    buf182 = buf180[1]
    del buf180
    buf183 = buf163; del buf163  # reuse
    buf184 = empty((128, ), device='cpu', dtype=torch.float32)
    buf185 = empty((128, ), device='cpu', dtype=torch.float32)
    buf186 = buf181; del buf181  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31(c_void_p(buf186.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del convolution_11
    del primals_23
    del relu_10
    del squeeze_34
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf187 = aten.convolution_backward(buf186, relu_9, primals_80, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf186
    del primals_80
    buf188 = buf187[0]
    buf189 = buf187[1]
    del buf187
    buf190 = buf184; del buf184  # reuse
    buf191 = empty((128, ), device='cpu', dtype=torch.float32)
    buf192 = empty((128, ), device='cpu', dtype=torch.float32)
    buf193 = buf188; del buf188  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf193.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del buf191
    del convolution_10
    del primals_21
    del relu_9
    del squeeze_31
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf194 = aten.convolution_backward(buf193, relu_8, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf193
    del primals_79
    buf195 = buf194[0]
    buf196 = buf194[1]
    del buf194
    buf197 = buf142; del buf142  # reuse
    buf198 = empty((256, ), device='cpu', dtype=torch.float32)
    buf199 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf200 = buf198; del buf198  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_33(c_void_p(buf200.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    del convolution_9
    del primals_19
    del squeeze_28
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf201 = aten.convolution_backward(buf199, relu_7, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_78
    buf202 = buf201[0]
    buf203 = buf201[1]
    del buf201
    buf204 = empty((64, ), device='cpu', dtype=torch.float32)
    buf205 = empty((64, ), device='cpu', dtype=torch.float32)
    buf206 = empty((64, ), device='cpu', dtype=torch.float32)
    buf207 = buf202; del buf202  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf207.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del convolution_8
    del primals_17
    del relu_7
    del squeeze_25
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf208 = aten.convolution_backward(buf207, relu_6, primals_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf207
    del primals_77
    buf209 = buf208[0]
    buf210 = buf208[1]
    del buf208
    buf211 = buf205; del buf205  # reuse
    buf212 = empty((64, ), device='cpu', dtype=torch.float32)
    buf213 = empty((64, ), device='cpu', dtype=torch.float32)
    buf214 = buf209; del buf209  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf214.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del convolution_7
    del primals_15
    del relu_6
    del squeeze_22
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf215 = aten.convolution_backward(buf214, relu_5, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf214
    del primals_76
    buf216 = buf215[0]
    buf217 = buf215[1]
    del buf215
    buf218 = buf175; del buf175  # reuse
    buf219 = empty((256, ), device='cpu', dtype=torch.float32)
    buf220 = empty((256, ), device='cpu', dtype=torch.float32)
    buf226 = empty((256, ), device='cpu', dtype=torch.float32)
    buf221 = empty((256, ), device='cpu', dtype=torch.float32)
    buf222 = buf199; del buf199  # reuse
    buf228 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf218.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()))
    del buf195
    del buf216
    del buf218
    del buf220
    del convolution_5
    del convolution_6
    del primals_11
    del primals_13
    del relu_5
    del relu_8
    del squeeze_19
    del unsqueeze_414
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf223 = aten.convolution_backward(buf222, getitem_6, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf222
    del primals_75
    buf224 = buf223[0]
    buf225 = buf223[1]
    del buf223
    buf227 = buf226; del buf226  # reuse
    cpp_fused_native_batch_norm_backward_37(c_void_p(buf227.data_ptr()), c_void_p(squeeze_16.data_ptr()))
    del squeeze_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf229 = aten.convolution_backward(buf228, relu_4, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf228
    del primals_74
    buf230 = buf229[0]
    buf231 = buf229[1]
    del buf229
    buf232 = buf212; del buf212  # reuse
    buf233 = empty((64, ), device='cpu', dtype=torch.float32)
    buf234 = empty((64, ), device='cpu', dtype=torch.float32)
    buf235 = buf230; del buf230  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(buf235.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    del convolution_4
    del primals_9
    del relu_4
    del squeeze_13
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf236 = aten.convolution_backward(buf235, relu_3, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf235
    del primals_73
    buf237 = buf236[0]
    buf238 = buf236[1]
    del buf236
    buf239 = buf233; del buf233  # reuse
    buf240 = empty((64, ), device='cpu', dtype=torch.float32)
    buf241 = empty((64, ), device='cpu', dtype=torch.float32)
    buf242 = buf237; del buf237  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf242.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del convolution_3
    del primals_7
    del relu_3
    del squeeze_10
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf243 = aten.convolution_backward(buf242, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf242
    del getitem_6
    del primals_72
    buf244 = buf243[0]
    buf245 = buf243[1]
    del buf243
    buf246 = buf224; del buf224  # reuse
    cpp_fused_add_40(c_void_p(buf246.data_ptr()), c_void_p(buf244.data_ptr()))
    del buf244
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf247 = aten.max_pool2d_with_indices_backward(buf246, relu_2, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7)
    del buf246
    del getitem_7
    buf248 = buf247
    del buf247
    buf249 = buf240; del buf240  # reuse
    buf250 = empty((64, ), device='cpu', dtype=torch.float32)
    buf251 = empty((64, ), device='cpu', dtype=torch.float32)
    buf252 = buf248; del buf248  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf252.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del buf250
    del convolution_2
    del primals_5
    del relu_2
    del squeeze_7
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf253 = aten.convolution_backward(buf252, relu_1, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf252
    del primals_71
    buf254 = buf253[0]
    buf255 = buf253[1]
    del buf253
    buf256 = empty((32, ), device='cpu', dtype=torch.float32)
    buf257 = empty((32, ), device='cpu', dtype=torch.float32)
    buf258 = empty((32, ), device='cpu', dtype=torch.float32)
    buf259 = buf254; del buf254  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42(c_void_p(buf259.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del buf257
    del convolution_1
    del primals_3
    del relu_1
    del squeeze_4
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf260 = aten.convolution_backward(buf259, relu, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf259
    del primals_70
    buf261 = buf260[0]
    buf262 = buf260[1]
    del buf260
    buf263 = empty((24, ), device='cpu', dtype=torch.float32)
    buf264 = empty((24, ), device='cpu', dtype=torch.float32)
    buf265 = empty((24, ), device='cpu', dtype=torch.float32)
    buf266 = buf261; del buf261  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf266.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del buf264
    del convolution
    del primals_1
    del relu
    del squeeze_1
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf267 = aten.convolution_backward(buf266, primals_195, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf266
    del primals_195
    del primals_69
    buf268 = buf267[1]
    return (buf265, buf263, buf258, buf256, buf251, buf249, buf241, buf239, buf234, buf232, buf227, buf219, buf221, buf219, buf213, buf211, buf206, buf204, buf200, buf197, buf192, buf190, buf185, buf183, buf178, buf170, buf172, buf170, buf164, buf162, buf157, buf155, buf151, buf148, buf143, buf141, buf136, buf134, buf129, buf121, buf123, buf121, buf115, buf113, reinterpret_tensor(buf104, (31, 64), (64, 1), 0), reinterpret_tensor(buf100, (31, 64), (64, 1), 0), buf93, buf91, buf87, buf84, buf79, buf77, reinterpret_tensor(buf68, (31, 128), (128, 1), 0), reinterpret_tensor(buf64, (31, 128), (128, 1), 0), buf56, buf54, buf49, buf39, buf42, buf39, buf34, buf32, reinterpret_tensor(buf23, (15, 128), (128, 1), 0), reinterpret_tensor(buf19, (15, 128), (128, 1), 0), buf12, buf10, buf5, buf3, buf268, buf262, buf255, buf245, buf238, buf231, buf225, buf217, buf210, buf203, buf196, buf189, buf182, buf176, buf168, buf161, buf154, buf147, buf140, buf133, buf127, buf119, buf112, buf90, buf83, buf76, buf53, buf46, buf38, buf31, buf9, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.int64)
    convolution_3 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((8192, 64), (64, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((8192, 64), (64, 1), device='cpu', dtype=torch.float32)
    bmm_1 = rand_strided((32, 256, 64), (16384, 64, 1), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((8192, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((8192, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((2048, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((2048, 128), (128, 1), device='cpu', dtype=torch.float32)
    bmm_5 = rand_strided((32, 64, 128), (8192, 128, 1), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    clone_21 = rand_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_25 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.bool)
    unsqueeze_126 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_30 = rand_strided((32, 64, 64), (4096, 1, 64), device='cpu', dtype=torch.float32)
    permute_31 = rand_strided((32, 128, 64), (8192, 64, 1), device='cpu', dtype=torch.float32)
    alias_36 = rand_strided((32, 64, 64), (4096, 64, 1), device='cpu', dtype=torch.float32)
    permute_35 = rand_strided((15, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_41 = rand_strided((15, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_43 = rand_strided((32, 128, 64), (8192, 64, 1), device='cpu', dtype=torch.float32)
    permute_44 = rand_strided((32, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_174 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_48 = rand_strided((32, 256, 256), (65536, 1, 256), device='cpu', dtype=torch.float32)
    permute_49 = rand_strided((32, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    alias_46 = rand_strided((32, 256, 256), (65536, 256, 1), device='cpu', dtype=torch.float32)
    permute_53 = rand_strided((31, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_59 = rand_strided((31, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_61 = rand_strided((32, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    permute_62 = rand_strided((32, 256, 128), (32768, 1, 256), device='cpu', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_66 = rand_strided((32, 256, 256), (65536, 1, 256), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((32, 64, 256), (16384, 256, 1), device='cpu', dtype=torch.float32)
    alias_56 = rand_strided((32, 256, 256), (65536, 256, 1), device='cpu', dtype=torch.float32)
    permute_71 = rand_strided((31, 64), (64, 1), device='cpu', dtype=torch.float32)
    permute_77 = rand_strided((31, 64), (64, 1), device='cpu', dtype=torch.float32)
    permute_79 = rand_strided((32, 64, 256), (16384, 256, 1), device='cpu', dtype=torch.float32)
    permute_80 = rand_strided((32, 256, 64), (16384, 1, 256), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_195, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, view_7, view_13, bmm_1, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, view_31, view_37, view_47, avg_pool2d, squeeze_76, relu_22, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_23, convolution_28, squeeze_85, relu_24, view_55, view_61, bmm_5, squeeze_88, relu_25, convolution_30, squeeze_91, clone_21, permute_25, le, unsqueeze_126, unsqueeze_138, permute_30, permute_31, alias_36, permute_35, permute_41, permute_43, permute_44, unsqueeze_150, unsqueeze_162, unsqueeze_174, unsqueeze_186, permute_48, permute_49, alias_46, permute_53, permute_59, permute_61, permute_62, unsqueeze_198, unsqueeze_210, unsqueeze_222, permute_66, permute_67, alias_56, permute_71, permute_77, permute_79, permute_80, unsqueeze_234, unsqueeze_246, unsqueeze_258, unsqueeze_270, unsqueeze_282, unsqueeze_294, unsqueeze_306, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('botnet26t_256', benchmark_compiled_module)
