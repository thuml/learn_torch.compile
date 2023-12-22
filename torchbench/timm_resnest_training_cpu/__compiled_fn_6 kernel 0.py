
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


cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr2 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1000L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2000L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3000L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(49.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(0.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                            auto tmp10 = tmp8 - tmp9;
                            auto tmp11 = tmp7 * tmp10;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp7 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp7;
                            tmp_acc1_vec = tmp_acc1_vec + tmp11;
                            tmp_acc2_vec = tmp_acc2_vec + tmp15;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(49.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp11.rsqrt();
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp16 + tmp10;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp7 * tmp20;
                        tmp15.store(out_ptr4 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        tmp21.store(out_ptr5 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_mul_sum_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (3584L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (25088L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (512L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (3584L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (25088L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (3584L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (25088L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (512L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (3584L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (25088L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            auto tmp12 = tmp11 / 9;
                            auto tmp13 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))));
                            auto tmp14 = tmp13 < tmp6;
                            auto tmp15 = tmp4 & tmp14;
                            auto tmp16 = decltype(tmp10)(tmp10 + tmp12);
                            auto tmp17 = tmp15 ? tmp16 : tmp10;
                            auto tmp19 = tmp18 / 9;
                            auto tmp20 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))));
                            auto tmp21 = tmp20 < tmp3;
                            auto tmp22 = tmp21 & tmp7;
                            auto tmp23 = decltype(tmp17)(tmp17 + tmp19);
                            auto tmp24 = tmp22 ? tmp23 : tmp17;
                            auto tmp26 = tmp25 / 9;
                            auto tmp27 = tmp21 & tmp14;
                            auto tmp28 = decltype(tmp24)(tmp24 + tmp26);
                            auto tmp29 = tmp27 ? tmp28 : tmp24;
                            out_ptr0[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (100352L*x0))] = tmp29;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x3_inner));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (1024L*x3) + (1024L*x3_inner) + (200704L*x0)));
                                    auto tmp3 = tmp1 * tmp2;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                }
                            }
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr0[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (1024L*x3) + (200704L*x0)));
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (1024L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(512L))) + (1024L*x0) + (c10::div_floor_integer((x1 + x1_inner), 512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((1024L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(512L))) + (1024L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(512L + (1024L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(512L))) + (1024L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        auto tmp10 = tmp1 * tmp9;
                        auto tmp11 = tmp2 - tmp10;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x0));
            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0));
            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x0));
            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0));
            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x0));
            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x0));
            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(256L + x0));
            auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(512L + x0));
            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(768L + x0));
            auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            auto tmp7 = to_float_mask(tmp6 <= tmp2);
            auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp7);
            auto tmp10 = tmp5 + tmp9;
            auto tmp12 = to_float_mask(tmp11 <= tmp2);
            auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp12);
            auto tmp15 = tmp10 + tmp14;
            auto tmp17 = to_float_mask(tmp16 <= tmp2);
            auto tmp19 = decltype(tmp2)::blendv(tmp18, tmp2, tmp17);
            auto tmp20 = tmp15 + tmp19;
            auto tmp23 = tmp21 - tmp22;
            auto tmp24 = tmp5 * tmp23;
            auto tmp26 = tmp25 - tmp22;
            auto tmp27 = tmp9 * tmp26;
            auto tmp28 = tmp24 + tmp27;
            auto tmp30 = tmp29 - tmp22;
            auto tmp31 = tmp14 * tmp30;
            auto tmp32 = tmp28 + tmp31;
            auto tmp34 = tmp33 - tmp22;
            auto tmp35 = tmp19 * tmp34;
            auto tmp36 = tmp32 + tmp35;
            auto tmp38 = static_cast<float>(1e-05);
            auto tmp39 = at::vec::Vectorized<float>(tmp38);
            auto tmp40 = tmp37 + tmp39;
            auto tmp41 = tmp40.rsqrt();
            auto tmp42 = tmp36 * tmp41;
            tmp20.store(out_ptr0 + static_cast<long>(x0));
            tmp42.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp12 = tmp10 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x2) + (1024L*x2_inner) + (200704L*x1)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + x2_inner + (196L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L))) + (100352L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L))) + (1024L*x1) + (static_cast<long>(c10::div_floor_integer(((512L*(c10::div_floor_integer((x0 + x0_inner), 512L))) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L))), 512L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>((512L*x1) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x2) + (1024L*x2_inner) + (200704L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = static_cast<float>(196.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 / tmp9;
                                auto tmp11 = tmp6 + tmp10;
                                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                                auto tmp15 = tmp13 - tmp14;
                                auto tmp16 = tmp12 * tmp15;
                                tmp_acc0_vec = tmp_acc0_vec + tmp12;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x2) + (200704L*x1)));
                            auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (196L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L))) + (100352L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L))) + (1024L*x1) + (static_cast<long>(c10::div_floor_integer(((512L*(c10::div_floor_integer((x0 + x0_inner), 512L))) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L))), 512L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>((512L*x1) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x2) + (200704L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = static_cast<float>(196.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 / tmp9;
                            auto tmp11 = tmp6 + tmp10;
                            auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                            auto tmp15 = tmp13 - tmp14;
                            auto tmp16 = tmp12 * tmp15;
                            tmp_acc0_vec = tmp_acc0_vec + tmp12;
                            tmp_acc1_vec = tmp_acc1_vec + tmp16;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (200704L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (196L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(512L))) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(512L))) + (1024L*x0) + (static_cast<long>(c10::div_floor_integer(((512L*(c10::div_floor_integer((x2 + x2_inner), 512L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(512L))), 512L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((512L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(196.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(1e-05);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(out_ptr2 + static_cast<long>(x2 + (1024L*x1) + (200704L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    float tmp_acc2 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x0 + (1024L*x3) + (14336L*x2) + (200704L*x1))];
                                auto tmp3 = in_ptr1[static_cast<long>(x0 + (1024L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x3, 2L))))))) + (1024L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x3, 2L)))))) >= 0L) ? 0L : 7L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 7L)) + (50176L*x1))];
                                auto tmp13 = in_ptr2[static_cast<long>(x0 + (1024L*x3) + (14336L*x2) + (200704L*x1))];
                                auto tmp16 = in_ptr3[static_cast<long>(x0 + (1024L*x3) + (14336L*x2) + (200704L*x1))];
                                auto tmp17 = in_ptr4[static_cast<long>(x0)];
                                auto tmp20 = in_ptr5[static_cast<long>(x0 + (1024L*x3) + (14336L*x2) + (200704L*x1))];
                                auto tmp21 = in_ptr6[static_cast<long>(x0)];
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = tmp0 <= tmp1;
                                auto tmp4 = tmp3 / 4;
                                auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                                auto tmp6 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))));
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                                auto tmp9 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x3, 2L))));
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = tmp7 & tmp10;
                                auto tmp12 = tmp11 ? tmp4 : tmp1;
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                auto tmp15 = tmp2 ? tmp1 : tmp14;
                                auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                                auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                                auto tmp22 = decltype(tmp20)(tmp20 - tmp21);
                                auto tmp23 = decltype(tmp15)(tmp15 * tmp22);
                                tmp_acc0 = tmp_acc0 + tmp15;
                                tmp_acc1 = tmp_acc1 + tmp19;
                                tmp_acc2 = tmp_acc2 + tmp23;
                            }
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                    out_ptr2[static_cast<long>(x0)] = tmp_acc2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (1024L*x2) + (14336L*x1) + (200704L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (1024L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (1024L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 7L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 7L)) + (50176L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x3 + (1024L*x2) + (14336L*x1) + (200704L*x0))];
                            auto tmp16 = in_ptr7[static_cast<long>(x3)];
                            auto tmp20 = in_ptr8[static_cast<long>(x3)];
                            auto tmp23 = in_ptr9[static_cast<long>(x3)];
                            auto tmp26 = in_ptr10[static_cast<long>(x3)];
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp4 = tmp3 / 4;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp9 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = tmp7 & tmp10;
                            auto tmp12 = tmp11 ? tmp4 : tmp1;
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = tmp2 ? tmp1 : tmp14;
                            auto tmp17 = static_cast<float>(1e-05);
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = 1 / std::sqrt(tmp18);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp15)(tmp15 * tmp21);
                            auto tmp24 = decltype(tmp23)(tmp23 + tmp17);
                            auto tmp25 = 1 / std::sqrt(tmp24);
                            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                            auto tmp28 = decltype(tmp15)(tmp15 * tmp27);
                            out_ptr3[static_cast<long>(x3 + (1024L*x2) + (14336L*x1) + (200704L*x0))] = tmp22;
                            out_ptr4[static_cast<long>(x3 + (1024L*x2) + (14336L*x1) + (200704L*x0))] = tmp28;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_mul_sum_8 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (3584L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (50176L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (3584L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (50176L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (3584L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (50176L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (3584L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (50176L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            auto tmp12 = tmp11 / 9;
                            auto tmp13 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))));
                            auto tmp14 = tmp13 < tmp6;
                            auto tmp15 = tmp4 & tmp14;
                            auto tmp16 = decltype(tmp10)(tmp10 + tmp12);
                            auto tmp17 = tmp15 ? tmp16 : tmp10;
                            auto tmp19 = tmp18 / 9;
                            auto tmp20 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))));
                            auto tmp21 = tmp20 < tmp3;
                            auto tmp22 = tmp21 & tmp7;
                            auto tmp23 = decltype(tmp17)(tmp17 + tmp19);
                            auto tmp24 = tmp22 ? tmp23 : tmp17;
                            auto tmp26 = tmp25 / 9;
                            auto tmp27 = tmp21 & tmp14;
                            auto tmp28 = decltype(tmp24)(tmp24 + tmp26);
                            auto tmp29 = tmp27 ? tmp28 : tmp24;
                            out_ptr0[static_cast<long>(x3 + (28L*x2) + (784L*x1) + (200704L*x0))] = tmp29;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(784L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x3 + (784L*x2) + (200704L*x0)), static_cast<long>(784L), tmp0, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x3_inner));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (512L*x3) + (512L*x3_inner) + (401408L*x0)));
                                    auto tmp3 = tmp1 * tmp2;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                }
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (512L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L))) + (512L*x0) + (c10::div_floor_integer((x1 + x1_inner), 256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((512L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L))) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(256L + (512L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L))) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        auto tmp10 = tmp1 * tmp9;
                        auto tmp11 = tmp2 - tmp10;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x0));
            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0));
            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x0));
            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0));
            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x0));
            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(384L + x0));
            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(128L + x0));
            auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(256L + x0));
            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(384L + x0));
            auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            auto tmp7 = to_float_mask(tmp6 <= tmp2);
            auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp7);
            auto tmp10 = tmp5 + tmp9;
            auto tmp12 = to_float_mask(tmp11 <= tmp2);
            auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp12);
            auto tmp15 = tmp10 + tmp14;
            auto tmp17 = to_float_mask(tmp16 <= tmp2);
            auto tmp19 = decltype(tmp2)::blendv(tmp18, tmp2, tmp17);
            auto tmp20 = tmp15 + tmp19;
            auto tmp23 = tmp21 - tmp22;
            auto tmp24 = tmp5 * tmp23;
            auto tmp26 = tmp25 - tmp22;
            auto tmp27 = tmp9 * tmp26;
            auto tmp28 = tmp24 + tmp27;
            auto tmp30 = tmp29 - tmp22;
            auto tmp31 = tmp14 * tmp30;
            auto tmp32 = tmp28 + tmp31;
            auto tmp34 = tmp33 - tmp22;
            auto tmp35 = tmp19 * tmp34;
            auto tmp36 = tmp32 + tmp35;
            auto tmp38 = static_cast<float>(1e-05);
            auto tmp39 = at::vec::Vectorized<float>(tmp38);
            auto tmp40 = tmp37 + tmp39;
            auto tmp41 = tmp40.rsqrt();
            auto tmp42 = tmp36 * tmp41;
            tmp20.store(out_ptr0 + static_cast<long>(x0));
            tmp42.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp12 = tmp10 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                        {
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x2) + (512L*x2_inner) + (401408L*x1)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + x2_inner + (784L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(256L))) + (200704L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(256L))) + (512L*x1) + (static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer((x0 + x0_inner), 256L))) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(256L))), 256L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>((256L*x1) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x2) + (512L*x2_inner) + (401408L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = static_cast<float>(784.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 / tmp9;
                                auto tmp11 = tmp6 + tmp10;
                                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                                auto tmp15 = tmp13 - tmp14;
                                auto tmp16 = tmp12 * tmp15;
                                tmp_acc0_vec = tmp_acc0_vec + tmp12;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                            }
                        }
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (784L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(256L))) + (200704L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(256L))) + (512L*x0) + (static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer((x2 + x2_inner), 256L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(256L))), 256L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((256L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(784.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(1e-05);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    float tmp_acc2 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x0 + (512L*x3) + (14336L*x2) + (401408L*x1))];
                                auto tmp3 = in_ptr1[static_cast<long>(x0 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x3, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x3, 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x1))];
                                auto tmp13 = in_ptr2[static_cast<long>(x0 + (512L*x3) + (14336L*x2) + (401408L*x1))];
                                auto tmp16 = in_ptr3[static_cast<long>(x0 + (512L*x3) + (14336L*x2) + (401408L*x1))];
                                auto tmp17 = in_ptr4[static_cast<long>(x0)];
                                auto tmp20 = in_ptr5[static_cast<long>(x0 + (512L*x3) + (14336L*x2) + (401408L*x1))];
                                auto tmp21 = in_ptr6[static_cast<long>(x0)];
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = tmp0 <= tmp1;
                                auto tmp4 = tmp3 / 4;
                                auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                                auto tmp6 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))));
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                                auto tmp9 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x3, 2L))));
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = tmp7 & tmp10;
                                auto tmp12 = tmp11 ? tmp4 : tmp1;
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                auto tmp15 = tmp2 ? tmp1 : tmp14;
                                auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                                auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                                auto tmp22 = decltype(tmp20)(tmp20 - tmp21);
                                auto tmp23 = decltype(tmp15)(tmp15 * tmp22);
                                tmp_acc0 = tmp_acc0 + tmp15;
                                tmp_acc1 = tmp_acc1 + tmp19;
                                tmp_acc2 = tmp_acc2 + tmp23;
                            }
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                    out_ptr2[static_cast<long>(x0)] = tmp_acc2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp16 = in_ptr7[static_cast<long>(x3)];
                            auto tmp20 = in_ptr8[static_cast<long>(x3)];
                            auto tmp23 = in_ptr9[static_cast<long>(x3)];
                            auto tmp26 = in_ptr10[static_cast<long>(x3)];
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp4 = tmp3 / 4;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp9 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = tmp7 & tmp10;
                            auto tmp12 = tmp11 ? tmp4 : tmp1;
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = tmp2 ? tmp1 : tmp14;
                            auto tmp17 = static_cast<float>(1e-05);
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = 1 / std::sqrt(tmp18);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp15)(tmp15 * tmp21);
                            auto tmp24 = decltype(tmp23)(tmp23 + tmp17);
                            auto tmp25 = 1 / std::sqrt(tmp24);
                            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                            auto tmp28 = decltype(tmp15)(tmp15 * tmp27);
                            out_ptr3[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))] = tmp22;
                            out_ptr4[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))] = tmp28;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_mul_sum_14 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (128L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (3584L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (100352L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (128L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (128L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (3584L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (100352L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (128L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (128L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (3584L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (100352L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (128L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (128L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (3584L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (3584L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (100352L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            auto tmp12 = tmp11 / 9;
                            auto tmp13 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))));
                            auto tmp14 = tmp13 < tmp6;
                            auto tmp15 = tmp4 & tmp14;
                            auto tmp16 = decltype(tmp10)(tmp10 + tmp12);
                            auto tmp17 = tmp15 ? tmp16 : tmp10;
                            auto tmp19 = tmp18 / 9;
                            auto tmp20 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))));
                            auto tmp21 = tmp20 < tmp3;
                            auto tmp22 = tmp21 & tmp7;
                            auto tmp23 = decltype(tmp17)(tmp17 + tmp19);
                            auto tmp24 = tmp22 ? tmp23 : tmp17;
                            auto tmp26 = tmp25 / 9;
                            auto tmp27 = tmp21 & tmp14;
                            auto tmp28 = decltype(tmp24)(tmp24 + tmp26);
                            auto tmp29 = tmp27 ? tmp28 : tmp24;
                            out_ptr0[static_cast<long>(x3 + (56L*x2) + (3136L*x1) + (401408L*x0))] = tmp29;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(3136L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x3 + (3136L*x2) + (401408L*x0)), static_cast<long>(3136L), tmp0, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x3_inner));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (256L*x3) + (256L*x3_inner) + (802816L*x0)));
                                    auto tmp3 = tmp1 * tmp2;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                }
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (256L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L))) + (256L*x0) + (c10::div_floor_integer((x1 + x1_inner), 128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>((256L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L))) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(128L + (256L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L))) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        auto tmp10 = tmp1 * tmp9;
                        auto tmp11 = tmp2 - tmp10;
                        tmp11.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x0));
            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0));
            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x0));
            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0));
            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(192L + x0));
            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x0));
            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x0));
            auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(128L + x0));
            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(192L + x0));
            auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            auto tmp7 = to_float_mask(tmp6 <= tmp2);
            auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp7);
            auto tmp10 = tmp5 + tmp9;
            auto tmp12 = to_float_mask(tmp11 <= tmp2);
            auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp12);
            auto tmp15 = tmp10 + tmp14;
            auto tmp17 = to_float_mask(tmp16 <= tmp2);
            auto tmp19 = decltype(tmp2)::blendv(tmp18, tmp2, tmp17);
            auto tmp20 = tmp15 + tmp19;
            auto tmp23 = tmp21 - tmp22;
            auto tmp24 = tmp5 * tmp23;
            auto tmp26 = tmp25 - tmp22;
            auto tmp27 = tmp9 * tmp26;
            auto tmp28 = tmp24 + tmp27;
            auto tmp30 = tmp29 - tmp22;
            auto tmp31 = tmp14 * tmp30;
            auto tmp32 = tmp28 + tmp31;
            auto tmp34 = tmp33 - tmp22;
            auto tmp35 = tmp19 * tmp34;
            auto tmp36 = tmp32 + tmp35;
            auto tmp38 = static_cast<float>(1e-05);
            auto tmp39 = at::vec::Vectorized<float>(tmp38);
            auto tmp40 = tmp37 + tmp39;
            auto tmp41 = tmp40.rsqrt();
            auto tmp42 = tmp36 * tmp41;
            tmp20.store(out_ptr0 + static_cast<long>(x0));
            tmp42.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp12 = tmp10 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                        {
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (256L*x2_inner) + (802816L*x1)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + x2_inner + (3136L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(128L))) + (401408L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(128L))) + (256L*x1) + (static_cast<long>(c10::div_floor_integer(((128L*(c10::div_floor_integer((x0 + x0_inner), 128L))) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(128L))), 128L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>((128L*x1) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x2) + (256L*x2_inner) + (802816L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = static_cast<float>(3136.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 / tmp9;
                                auto tmp11 = tmp6 + tmp10;
                                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                                auto tmp15 = tmp13 - tmp14;
                                auto tmp16 = tmp12 * tmp15;
                                tmp_acc0_vec = tmp_acc0_vec + tmp12;
                                tmp_acc1_vec = tmp_acc1_vec + tmp16;
                            }
                        }
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (3136L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L))) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L))) + (256L*x0) + (static_cast<long>(c10::div_floor_integer(((128L*(c10::div_floor_integer((x2 + x2_inner), 128L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L))), 128L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((128L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(3136.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(1e-05);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    float tmp_acc2 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x0 + (256L*x3) + (14336L*x2) + (802816L*x1))];
                                auto tmp3 = in_ptr1[static_cast<long>(x0 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x3, 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x3, 2L)))))) >= 0L) ? 0L : 28L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 28L)) + (200704L*x1))];
                                auto tmp13 = in_ptr2[static_cast<long>(x0 + (256L*x3) + (14336L*x2) + (802816L*x1))];
                                auto tmp16 = in_ptr3[static_cast<long>(x0 + (256L*x3) + (14336L*x2) + (802816L*x1))];
                                auto tmp17 = in_ptr4[static_cast<long>(x0)];
                                auto tmp20 = in_ptr5[static_cast<long>(x0 + (256L*x3) + (14336L*x2) + (802816L*x1))];
                                auto tmp21 = in_ptr6[static_cast<long>(x0)];
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = tmp0 <= tmp1;
                                auto tmp4 = tmp3 / 4;
                                auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                                auto tmp6 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer(x2, 2L))));
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                                auto tmp9 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer(x3, 2L))));
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = tmp7 & tmp10;
                                auto tmp12 = tmp11 ? tmp4 : tmp1;
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                auto tmp15 = tmp2 ? tmp1 : tmp14;
                                auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                                auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                                auto tmp22 = decltype(tmp20)(tmp20 - tmp21);
                                auto tmp23 = decltype(tmp15)(tmp15 * tmp22);
                                tmp_acc0 = tmp_acc0 + tmp15;
                                tmp_acc1 = tmp_acc1 + tmp19;
                                tmp_acc2 = tmp_acc2 + tmp23;
                            }
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                    out_ptr2[static_cast<long>(x0)] = tmp_acc2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 28L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 28L)) + (200704L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0))];
                            auto tmp16 = in_ptr7[static_cast<long>(x3)];
                            auto tmp20 = in_ptr8[static_cast<long>(x3)];
                            auto tmp23 = in_ptr9[static_cast<long>(x3)];
                            auto tmp26 = in_ptr10[static_cast<long>(x3)];
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp4 = tmp3 / 4;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp9 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = tmp7 & tmp10;
                            auto tmp12 = tmp11 ? tmp4 : tmp1;
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = tmp2 ? tmp1 : tmp14;
                            auto tmp17 = static_cast<float>(1e-05);
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = 1 / std::sqrt(tmp18);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp15)(tmp15 * tmp21);
                            auto tmp24 = decltype(tmp23)(tmp23 + tmp17);
                            auto tmp25 = 1 / std::sqrt(tmp24);
                            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                            auto tmp28 = decltype(tmp15)(tmp15 * tmp27);
                            out_ptr3[static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0))] = tmp22;
                            out_ptr4[static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0))] = tmp28;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sum_20 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(3136L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x3) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (128L*x3) + (401408L*x0)));
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (128L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L))) + (128L*x0) + (c10::div_floor_integer((x1 + x1_inner), 64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((128L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L))) + (128L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(64L + (128L*x0) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L))) + (128L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp9 = tmp5 + tmp8;
                        auto tmp10 = tmp1 * tmp9;
                        auto tmp11 = tmp2 - tmp10;
                        tmp11.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(32L + x0));
            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x0));
            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x0));
            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0));
            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(96L + x0));
            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(96L + x0));
            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(32L + x0));
            auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x0));
            auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(96L + x0));
            auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            auto tmp7 = to_float_mask(tmp6 <= tmp2);
            auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp7);
            auto tmp10 = tmp5 + tmp9;
            auto tmp12 = to_float_mask(tmp11 <= tmp2);
            auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp12);
            auto tmp15 = tmp10 + tmp14;
            auto tmp17 = to_float_mask(tmp16 <= tmp2);
            auto tmp19 = decltype(tmp2)::blendv(tmp18, tmp2, tmp17);
            auto tmp20 = tmp15 + tmp19;
            auto tmp23 = tmp21 - tmp22;
            auto tmp24 = tmp5 * tmp23;
            auto tmp26 = tmp25 - tmp22;
            auto tmp27 = tmp9 * tmp26;
            auto tmp28 = tmp24 + tmp27;
            auto tmp30 = tmp29 - tmp22;
            auto tmp31 = tmp14 * tmp30;
            auto tmp32 = tmp28 + tmp31;
            auto tmp34 = tmp33 - tmp22;
            auto tmp35 = tmp19 * tmp34;
            auto tmp36 = tmp32 + tmp35;
            auto tmp38 = static_cast<float>(1e-05);
            auto tmp39 = at::vec::Vectorized<float>(tmp38);
            auto tmp40 = tmp37 + tmp39;
            auto tmp41 = tmp40.rsqrt();
            auto tmp42 = tmp36 * tmp41;
            tmp20.store(out_ptr0 + static_cast<long>(x0));
            tmp42.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp12 = tmp10 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x2) + (401408L*x1)));
                            auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*x2) + (200704L*x1) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x0 + x0_inner)) % static_cast<long>(64L))) + (128L*x1) + (static_cast<long>(c10::div_floor_integer(((64L*(c10::div_floor_integer((x0 + x0_inner), 64L))) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(64L))), 64L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>((64L*x1) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x2) + (401408L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = static_cast<float>(3136.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 / tmp9;
                            auto tmp11 = tmp6 + tmp10;
                            auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                            auto tmp15 = tmp13 - tmp14;
                            auto tmp16 = tmp12 * tmp15;
                            tmp_acc0_vec = tmp_acc0_vec + tmp12;
                            tmp_acc1_vec = tmp_acc1_vec + tmp16;
                        }
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((64L*x1) + (200704L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((2L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L))) + (128L*x0) + (static_cast<long>(c10::div_floor_integer(((64L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L))), 64L)) % static_cast<long>(2L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((64L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(3136.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(1e-05);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_54, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_72, primals_74, primals_76, primals_77, primals_79, primals_80, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, getitem, getitem_1, convolution_3, relu_3, convolution_4, relu_4, mean, convolution_5, relu_5, div, sum_3, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, mean_1, convolution_11, relu_9, div_1, sum_6, avg_pool2d, convolution_13, avg_pool2d_1, convolution_14, relu_10, convolution_15, relu_11, convolution_16, relu_12, mean_2, convolution_17, relu_13, div_2, sum_9, avg_pool2d_2, convolution_19, avg_pool2d_3, convolution_20, relu_14, convolution_21, relu_15, convolution_22, relu_16, mean_3, convolution_23, relu_17, div_3, sum_12, avg_pool2d_4, convolution_25, avg_pool2d_5, convolution_26, view_24, permute_5, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (128, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_16, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_20, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_28, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_38, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (512, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_52, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_56, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_58, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_61, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_62, (1024, ), (1, ))
    assert_size_stride(primals_64, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (1024, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_70, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_74, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_76, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_77, (2048, ), (1, ))
    assert_size_stride(primals_79, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_80, (2048, ), (1, ))
    assert_size_stride(primals_84, (32, ), (1, ))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_87, (32, ), (1, ))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, ), (1, ))
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (512, ), (1, ))
    assert_size_stride(primals_141, (1024, ), (1, ))
    assert_size_stride(primals_142, (1024, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_147, (2048, ), (1, ))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (2048, ), (1, ))
    assert_size_stride(primals_153, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(relu, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(relu_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu_2, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_1, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_3, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(relu_3, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_4, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_4, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(mean, (4, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_5, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(relu_5, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div, (4, 2, 1, 64), (128, 1, 128, 2))
    assert_size_stride(sum_3, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_7, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_8, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_6, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_9, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_7, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_10, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_8, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(mean_1, (4, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_11, (4, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(relu_9, (4, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(div_1, (4, 2, 1, 128), (256, 1, 256, 2))
    assert_size_stride(sum_6, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(avg_pool2d, (4, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_13, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(avg_pool2d_1, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_14, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_10, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_15, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_11, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_16, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_12, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(mean_2, (4, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_17, (4, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(relu_13, (4, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(div_2, (4, 2, 1, 256), (512, 1, 512, 2))
    assert_size_stride(sum_9, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(avg_pool2d_2, (4, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_19, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(avg_pool2d_3, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_20, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_14, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_21, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_15, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_22, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_16, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(mean_3, (4, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_23, (4, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu_17, (4, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(div_3, (4, 2, 1, 512), (1024, 1, 1024, 2))
    assert_size_stride(sum_12, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(avg_pool2d_4, (4, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_25, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(avg_pool2d_5, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_26, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(view_24, (4, 2048), (2048, 1))
    assert_size_stride(permute_5, (1000, 2048), (2048, 1))
    assert_size_stride(le, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_5, out=buf0)
    del permute_5
    buf1 = empty((1000, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), view_24, out=buf1)
    del view_24
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf10 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0(c_void_p(buf5.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf12.data_ptr()))
    del buf0
    del convolution_25
    del convolution_26
    del le
    del primals_147
    del primals_150
    del primals_151
    del primals_77
    del primals_80
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, avg_pool2d_5, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_5
    del buf6
    del primals_79
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf11 = buf10; del buf10  # reuse
    cpp_fused_native_batch_norm_backward_1(c_void_p(buf11.data_ptr()), c_void_p(primals_148.data_ptr()))
    del primals_148
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf13 = aten.convolution_backward(buf12, avg_pool2d_4, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_4
    del primals_76
    buf14 = buf13[0]
    buf15 = buf13[1]
    del buf13
    buf16 = reinterpret_tensor(buf12, (4, 512, 14, 14), (100352, 196, 14, 1), 0); del buf12  # reuse
    buf17 = empty((4, 2, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf18 = empty((4, 1024, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_backward_convolution_backward_mul_sum_2(c_void_p(buf14.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del buf14
    del buf17
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf19 = aten.convolution_backward(buf18, relu_17, primals_74, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf18
    del primals_74
    buf20 = buf19[0]
    buf21 = buf19[1]
    buf22 = buf19[2]
    del buf19
    buf23 = empty((256, ), device='cpu', dtype=torch.float32)
    buf24 = empty((256, ), device='cpu', dtype=torch.float32)
    buf25 = buf24; del buf24  # reuse
    buf26 = buf20; del buf20  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf23.data_ptr()))
    del convolution_23
    del primals_144
    del primals_145
    del primals_72
    del relu_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf27 = aten.convolution_backward(buf26, mean_3, primals_70, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_3
    del primals_70
    buf28 = buf27[0]
    buf29 = buf27[1]
    buf30 = buf27[2]
    del buf27
    buf31 = reinterpret_tensor(buf26, (1024, ), (1, ), 0); del buf26  # reuse
    buf32 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf33 = buf32; del buf32  # reuse
    buf34 = empty_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf33.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf34.data_ptr()))
    del buf16
    del convolution_22
    del div_3
    del primals_141
    del primals_142
    del primals_68
    del relu_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf35 = aten.convolution_backward(buf34, relu_15, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
    del primals_67
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = empty((512, ), device='cpu', dtype=torch.float32)
    buf39 = empty((512, ), device='cpu', dtype=torch.float32)
    buf40 = buf39; del buf39  # reuse
    buf41 = buf36; del buf36  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_21
    del primals_138
    del primals_139
    del primals_65
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf42 = aten.convolution_backward(buf41, relu_14, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf41
    del primals_64
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf46 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf52 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf47 = buf46; del buf46  # reuse
    buf48 = buf34; del buf34  # reuse
    buf54 = empty_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf47.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf54.data_ptr()))
    del buf43
    del buf8
    del convolution_19
    del convolution_20
    del primals_132
    del primals_135
    del primals_136
    del primals_59
    del primals_62
    del relu_14
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf49 = aten.convolution_backward(buf48, avg_pool2d_3, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_3
    del buf48
    del primals_61
    buf50 = buf49[0]
    buf51 = buf49[1]
    del buf49
    buf53 = buf52; del buf52  # reuse
    cpp_fused_native_batch_norm_backward_7(c_void_p(buf53.data_ptr()), c_void_p(primals_133.data_ptr()))
    del primals_133
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf55 = aten.convolution_backward(buf54, avg_pool2d_2, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_2
    del primals_58
    buf56 = buf55[0]
    buf57 = buf55[1]
    del buf55
    buf58 = reinterpret_tensor(buf54, (4, 256, 28, 28), (200704, 784, 28, 1), 0); del buf54  # reuse
    buf59 = reinterpret_tensor(buf28, (4, 2, 256, 1, 1), (512, 256, 1, 1, 1), 0); del buf28  # reuse
    buf60 = empty((4, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_backward_convolution_backward_mul_sum_8(c_void_p(buf56.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del buf56
    del buf59
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf61 = aten.convolution_backward(buf60, relu_13, primals_56, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf60
    del primals_56
    buf62 = buf61[0]
    buf63 = buf61[1]
    buf64 = buf61[2]
    del buf61
    buf65 = empty((128, ), device='cpu', dtype=torch.float32)
    buf66 = empty((128, ), device='cpu', dtype=torch.float32)
    buf67 = buf66; del buf66  # reuse
    buf68 = buf62; del buf62  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf65.data_ptr()))
    del convolution_17
    del primals_129
    del primals_130
    del primals_54
    del relu_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf69 = aten.convolution_backward(buf68, mean_2, primals_52, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_2
    del primals_52
    buf70 = buf69[0]
    buf71 = buf69[1]
    buf72 = buf69[2]
    del buf69
    buf73 = reinterpret_tensor(buf68, (512, ), (1, ), 0); del buf68  # reuse
    buf74 = empty((512, ), device='cpu', dtype=torch.float32)
    buf75 = buf74; del buf74  # reuse
    buf76 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf75.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()))
    del buf58
    del convolution_16
    del div_2
    del primals_126
    del primals_127
    del primals_50
    del relu_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf77 = aten.convolution_backward(buf76, relu_11, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
    del primals_49
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf80 = empty((256, ), device='cpu', dtype=torch.float32)
    buf81 = empty((256, ), device='cpu', dtype=torch.float32)
    buf82 = buf81; del buf81  # reuse
    buf83 = buf78; del buf78  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf80.data_ptr()))
    del convolution_15
    del primals_123
    del primals_124
    del primals_47
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf84 = aten.convolution_backward(buf83, relu_10, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf83
    del primals_46
    buf85 = buf84[0]
    buf86 = buf84[1]
    del buf84
    buf87 = empty((512, ), device='cpu', dtype=torch.float32)
    buf88 = empty((512, ), device='cpu', dtype=torch.float32)
    buf94 = empty((512, ), device='cpu', dtype=torch.float32)
    buf89 = buf88; del buf88  # reuse
    buf90 = buf76; del buf76  # reuse
    buf96 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf89.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf96.data_ptr()))
    del buf50
    del buf85
    del convolution_13
    del convolution_14
    del primals_117
    del primals_120
    del primals_121
    del primals_41
    del primals_44
    del relu_10
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf91 = aten.convolution_backward(buf90, avg_pool2d_1, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_1
    del buf90
    del primals_43
    buf92 = buf91[0]
    buf93 = buf91[1]
    del buf91
    buf95 = buf94; del buf94  # reuse
    cpp_fused_native_batch_norm_backward_13(c_void_p(buf95.data_ptr()), c_void_p(primals_118.data_ptr()))
    del primals_118
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf97 = aten.convolution_backward(buf96, avg_pool2d, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d
    del primals_40
    buf98 = buf97[0]
    buf99 = buf97[1]
    del buf97
    buf100 = reinterpret_tensor(buf96, (4, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf96  # reuse
    buf101 = reinterpret_tensor(buf70, (4, 2, 128, 1, 1), (256, 128, 1, 1, 1), 0); del buf70  # reuse
    buf102 = empty((4, 256, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_backward_convolution_backward_mul_sum_14(c_void_p(buf98.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del buf101
    del buf98
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf103 = aten.convolution_backward(buf102, relu_9, primals_38, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf102
    del primals_38
    buf104 = buf103[0]
    buf105 = buf103[1]
    buf106 = buf103[2]
    del buf103
    buf107 = empty((64, ), device='cpu', dtype=torch.float32)
    buf108 = empty((64, ), device='cpu', dtype=torch.float32)
    buf109 = buf108; del buf108  # reuse
    buf110 = buf104; del buf104  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf107.data_ptr()))
    del convolution_11
    del primals_114
    del primals_115
    del primals_36
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf111 = aten.convolution_backward(buf110, mean_1, primals_34, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_1
    del primals_34
    buf112 = buf111[0]
    buf113 = buf111[1]
    buf114 = buf111[2]
    del buf111
    buf115 = reinterpret_tensor(buf110, (256, ), (1, ), 0); del buf110  # reuse
    buf116 = empty((256, ), device='cpu', dtype=torch.float32)
    buf117 = buf116; del buf116  # reuse
    buf118 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf117.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()))
    del buf100
    del convolution_10
    del div_1
    del primals_111
    del primals_112
    del primals_32
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf119 = aten.convolution_backward(buf118, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
    del primals_31
    buf120 = buf119[0]
    buf121 = buf119[1]
    del buf119
    buf122 = empty((128, ), device='cpu', dtype=torch.float32)
    buf123 = empty((128, ), device='cpu', dtype=torch.float32)
    buf124 = buf123; del buf123  # reuse
    buf125 = buf120; del buf120  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf122.data_ptr()))
    del convolution_9
    del primals_108
    del primals_109
    del primals_29
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf126 = aten.convolution_backward(buf125, relu_6, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_28
    buf127 = buf126[0]
    buf128 = buf126[1]
    del buf126
    buf129 = empty((256, ), device='cpu', dtype=torch.float32)
    buf130 = empty((256, ), device='cpu', dtype=torch.float32)
    buf136 = empty((256, ), device='cpu', dtype=torch.float32)
    buf131 = buf130; del buf130  # reuse
    buf132 = buf118; del buf118  # reuse
    buf138 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf131.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf138.data_ptr()))
    del buf127
    del buf92
    del convolution_7
    del convolution_8
    del primals_102
    del primals_105
    del primals_106
    del primals_23
    del primals_26
    del relu_6
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf133 = aten.convolution_backward(buf132, getitem, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf132
    del primals_25
    buf134 = buf133[0]
    buf135 = buf133[1]
    del buf133
    buf137 = buf136; del buf136  # reuse
    cpp_fused_native_batch_norm_backward_19(c_void_p(buf137.data_ptr()), c_void_p(primals_103.data_ptr()))
    del primals_103
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf139 = aten.convolution_backward(buf138, sum_3, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf138
    del primals_22
    del sum_3
    buf140 = buf139[0]
    buf141 = buf139[1]
    del buf139
    buf142 = reinterpret_tensor(buf112, (4, 2, 64, 1, 1), (128, 64, 1, 1, 1), 0); del buf112  # reuse
    buf143 = empty((4, 128, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_mul_sum_20(c_void_p(buf140.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(div.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    del buf142
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf144 = aten.convolution_backward(buf143, relu_5, primals_20, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf143
    del primals_20
    buf145 = buf144[0]
    buf146 = buf144[1]
    buf147 = buf144[2]
    del buf144
    buf148 = empty((32, ), device='cpu', dtype=torch.float32)
    buf149 = empty((32, ), device='cpu', dtype=torch.float32)
    buf150 = buf149; del buf149  # reuse
    buf151 = buf145; del buf145  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf148.data_ptr()))
    del convolution_5
    del primals_100
    del primals_18
    del primals_99
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf152 = aten.convolution_backward(buf151, mean, primals_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean
    del primals_16
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    del buf152
    buf156 = reinterpret_tensor(buf151, (128, ), (1, ), 0); del buf151  # reuse
    buf157 = empty((128, ), device='cpu', dtype=torch.float32)
    buf158 = buf157; del buf157  # reuse
    buf159 = buf125; del buf125  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf158.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(div.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf159.data_ptr()))
    del buf140
    del buf153
    del convolution_4
    del div
    del primals_14
    del primals_96
    del primals_97
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf160 = aten.convolution_backward(buf159, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
    del buf159
    del primals_13
    buf161 = buf160[0]
    buf162 = buf160[1]
    del buf160
    buf163 = empty((64, ), device='cpu', dtype=torch.float32)
    buf164 = empty((64, ), device='cpu', dtype=torch.float32)
    buf165 = buf164; del buf164  # reuse
    buf166 = buf161; del buf161  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf163.data_ptr()))
    del convolution_3
    del primals_11
    del primals_93
    del primals_94
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf167 = aten.convolution_backward(buf166, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf166
    del getitem
    del primals_10
    buf168 = buf167[0]
    buf169 = buf167[1]
    del buf167
    buf170 = buf134; del buf134  # reuse
    cpp_fused_add_24(c_void_p(buf170.data_ptr()), c_void_p(buf168.data_ptr()))
    del buf168
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf171 = aten.max_pool2d_with_indices_backward(buf170, relu_2, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1)
    del buf170
    del getitem_1
    buf172 = buf171
    del buf171
    buf173 = empty((64, ), device='cpu', dtype=torch.float32)
    buf174 = empty((64, ), device='cpu', dtype=torch.float32)
    buf175 = buf174; del buf174  # reuse
    buf176 = buf172; del buf172  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf173.data_ptr()))
    del convolution_2
    del primals_8
    del primals_90
    del primals_91
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf177 = aten.convolution_backward(buf176, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf176
    del primals_7
    buf178 = buf177[0]
    buf179 = buf177[1]
    del buf177
    buf180 = empty((32, ), device='cpu', dtype=torch.float32)
    buf181 = empty((32, ), device='cpu', dtype=torch.float32)
    buf182 = buf181; del buf181  # reuse
    buf183 = buf178; del buf178  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf180.data_ptr()))
    del convolution_1
    del primals_5
    del primals_87
    del primals_88
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf184 = aten.convolution_backward(buf183, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf183
    del primals_4
    buf185 = buf184[0]
    buf186 = buf184[1]
    del buf184
    buf187 = empty((32, ), device='cpu', dtype=torch.float32)
    buf188 = empty((32, ), device='cpu', dtype=torch.float32)
    buf189 = buf188; del buf188  # reuse
    buf190 = buf185; del buf185  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf187.data_ptr()))
    del convolution
    del primals_2
    del primals_84
    del primals_85
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf191 = aten.convolution_backward(buf190, primals_153, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf190
    del primals_1
    del primals_153
    buf192 = buf191[1]
    return (buf192, buf189, buf187, buf186, buf182, buf180, buf179, buf175, buf173, buf169, buf165, buf163, buf162, buf158, buf156, buf154, buf155, buf150, buf148, buf146, buf147, buf141, buf137, buf129, buf135, buf131, buf129, buf128, buf124, buf122, buf121, buf117, buf115, buf113, buf114, buf109, buf107, buf105, buf106, buf99, buf95, buf87, buf93, buf89, buf87, buf86, buf82, buf80, buf79, buf75, buf73, buf71, buf72, buf67, buf65, buf63, buf64, buf57, buf53, buf45, buf51, buf47, buf45, buf44, buf40, buf38, buf37, buf33, buf31, buf29, buf30, buf25, buf23, buf21, buf22, buf15, buf11, buf3, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((512, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1024, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    getitem = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.int64)
    convolution_3 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    mean = rand_strided((4, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div = rand_strided((4, 2, 1, 64), (128, 1, 128, 2), device='cpu', dtype=torch.float32)
    sum_3 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((4, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((4, 2, 1, 128), (256, 1, 256, 2), device='cpu', dtype=torch.float32)
    sum_6 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    avg_pool2d = rand_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((4, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((4, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((4, 2, 1, 256), (512, 1, 512, 2), device='cpu', dtype=torch.float32)
    sum_9 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((4, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    avg_pool2d_3 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((4, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((4, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((4, 2, 1, 512), (1024, 1, 1024, 2), device='cpu', dtype=torch.float32)
    sum_12 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    avg_pool2d_4 = rand_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    avg_pool2d_5 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((4, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_5 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_54, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_72, primals_74, primals_76, primals_77, primals_79, primals_80, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, getitem, getitem_1, convolution_3, relu_3, convolution_4, relu_4, mean, convolution_5, relu_5, div, sum_3, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, mean_1, convolution_11, relu_9, div_1, sum_6, avg_pool2d, convolution_13, avg_pool2d_1, convolution_14, relu_10, convolution_15, relu_11, convolution_16, relu_12, mean_2, convolution_17, relu_13, div_2, sum_9, avg_pool2d_2, convolution_19, avg_pool2d_3, convolution_20, relu_14, convolution_21, relu_15, convolution_22, relu_16, mean_3, convolution_23, relu_17, div_3, sum_12, avg_pool2d_4, convolution_25, avg_pool2d_5, convolution_26, view_24, permute_5, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_resnest', benchmark_compiled_module)
