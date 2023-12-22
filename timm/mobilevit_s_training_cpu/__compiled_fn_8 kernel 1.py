
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


cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x2) + (40960L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x2) + (40960L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(64.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp9 = tmp5 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(8L))
                    {
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (640L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (640L*x1) + (640L*x1_inner) + (40960L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (640L*x1) + (640L*x1_inner) + (40960L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp1 = static_cast<float>(64.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
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
                            tmp23.store(tmp24 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, out_ptr4 + static_cast<long>(x1 + (64L*x2) + (40960L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.001953125);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_2 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(160L + x0 + (320L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(160L + x1 + (320L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.001953125);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (32L*x0))] = static_cast<float>(tmp_acc0);
                    }
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
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0);
                    }
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x1 + (16L*x0))];
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x2) + (64L*x2_inner) + (15360L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp7 = out_ptr0[static_cast<long>(x0 + (32L*x1))];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = static_cast<float>(240.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp9 - tmp13;
                        auto tmp15 = at::vec::Vectorized<float>(tmp0);
                        auto tmp16 = tmp15 * tmp14;
                        tmp16.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (3840L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (64L*x0) + (64L*x0_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (64L*x0) + (64L*x0_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (64L*x0) + (64L*x0_inner) + (15360L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x2) + (3840L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((240L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(8L))) + (1920L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x0) + (64L*x0_inner) + (15360L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 8L)) % static_cast<long>(8L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x0) + (64L*x0_inner) + (15360L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (8L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (16L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (64L*x0) + (64L*x0_inner) + (15360L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 16L)) % static_cast<long>(7680L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(240.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-122880L) + x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-245760L) + x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (240L*x0) + (720L*x2) + (11520L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (720L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(240.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(240.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-122880L) + x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-245760L) + x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (240L*x0) + (720L*x2) + (11520L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (720L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(240.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(245760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(240.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-122880L) + x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-245760L) + x3 + (240L*x2) + (3840L*x1) + (122880L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (240L*x0) + (720L*x2) + (11520L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_layer_norm_backward_sum_18 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (720L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((240L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(16L))) + (3840L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (15360L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr5[static_cast<long>((16L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (64L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (15360L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((240L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(16L))) + (3840L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (15360L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((16L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (64L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (15360L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((240L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(16L))) + (3840L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (15360L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (15360L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 64L)) % static_cast<long>(240L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr2[static_cast<long>((16L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (64L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (64L*x3) + (15360L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 15360L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (16L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 16L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = static_cast<float>(240.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = tmp9 - tmp12;
                            auto tmp14 = tmp1 * tmp13;
                            auto tmp15 = tmp0 + tmp14;
                            tmp15.store(out_ptr5 + static_cast<long>(x2 + (8L*x1) + (64L*x3) + (15360L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.001953125);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_20 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (320L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.001953125);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.001953125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00048828125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00048828125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00048828125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (32L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (64L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x1 + (64L*x0))];
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x2) + (256L*x2_inner) + (49152L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp7 = out_ptr0[static_cast<long>(x0 + (32L*x1))];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (64L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = static_cast<float>(192.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp9 - tmp13;
                        auto tmp15 = at::vec::Vectorized<float>(tmp0);
                        auto tmp16 = tmp15 * tmp14;
                        tmp16.store(out_ptr2 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (256L*x0) + (256L*x0_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (256L*x0) + (256L*x0_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (256L*x0) + (256L*x0_inner) + (49152L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x2) + (12288L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((192L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(16L))) + (3072L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x0) + (256L*x0_inner) + (49152L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 16L)) % static_cast<long>(16L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x0) + (256L*x0_inner) + (49152L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(8L))) + (16L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (32L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (256L*x0) + (256L*x0_inner) + (49152L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 32L)) % static_cast<long>(12288L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(192.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-393216L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-786432L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (192L*x0) + (576L*x2) + (36864L*x1)));
                        }
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(192.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(192.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-393216L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-786432L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (192L*x0) + (576L*x2) + (36864L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(192.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(192.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-393216L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-786432L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (192L*x0) + (576L*x2) + (36864L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(192.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(192.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-393216L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-786432L) + x3 + (192L*x2) + (12288L*x1) + (393216L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (192L*x0) + (576L*x2) + (36864L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_layer_norm_backward_sum_45 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((192L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(64L))) + (12288L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (49152L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr5[static_cast<long>((64L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (256L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (49152L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((192L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(64L))) + (12288L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (49152L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((64L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (256L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (49152L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((192L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(64L))) + (12288L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (49152L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (49152L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 256L)) % static_cast<long>(192L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr2[static_cast<long>((64L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (256L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (256L*x3) + (49152L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 49152L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (32L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 32L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = static_cast<float>(192.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = tmp9 - tmp12;
                            auto tmp14 = tmp1 * tmp13;
                            auto tmp15 = tmp0 + tmp14;
                            tmp15.store(out_ptr5 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (49152L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00048828125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00048828125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00048828125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(96L + x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(96L + x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x0) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (32L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x1 + (256L*x0))];
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x0) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x2) + (1024L*x2_inner) + (147456L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp7 = out_ptr0[static_cast<long>(x0 + (32L*x1))];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = static_cast<float>(144.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp14 = tmp9 - tmp13;
                        auto tmp15 = at::vec::Vectorized<float>(tmp0);
                        auto tmp16 = tmp15 * tmp14;
                        tmp16.store(out_ptr2 + static_cast<long>(x2 + (144L*x1) + (36864L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (1024L*x0) + (1024L*x0_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (1024L*x0) + (1024L*x0_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x2) + (1024L*x0) + (1024L*x0_inner) + (147456L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x2) + (36864L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((144L*(static_cast<long>(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L)))) % static_cast<long>(32L))) + (4608L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x0) + (1024L*x0_inner) + (147456L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 32L)) % static_cast<long>(32L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x0) + (1024L*x0_inner) + (147456L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (static_cast<long>(x2) % static_cast<long>(4L))), 4L)) % static_cast<long>(16L))) + (32L*(c10::div_floor_integer((static_cast<long>(x2) % static_cast<long>(4L)), 2L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*x1) + (1024L*x0) + (1024L*x0_inner) + (147456L*(c10::div_floor_integer(x2, 4L))) + (static_cast<long>(x2) % static_cast<long>(4L))), 64L)) % static_cast<long>(18432L))) + (static_cast<long>((static_cast<long>(x2) % static_cast<long>(4L))) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (288L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(144.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (144L*x2) + (36864L*x1) + (1179648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1179648L) + x3 + (144L*x2) + (36864L*x1) + (1179648L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2359296L) + x3 + (144L*x2) + (36864L*x1) + (1179648L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (144L*x0) + (432L*x2) + (110592L*x1)));
                        }
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (432L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(144.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_fill_mul_silu_sub_sum_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (288L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(144.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (32L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(32);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (144L*x2) + (36864L*x1) + (1179648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(64);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1179648L) + x3 + (144L*x2) + (36864L*x1) + (1179648L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(96);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2359296L) + x3 + (144L*x2) + (36864L*x1) + (1179648L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (144L*x0) + (432L*x2) + (110592L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_layer_norm_backward_sum_62 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (432L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((144L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(256L))) + (36864L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (147456L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr5[static_cast<long>((256L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (1024L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (147456L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((144L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(256L))) + (36864L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (147456L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((256L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (1024L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (147456L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr3[static_cast<long>((144L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(256L))) + (36864L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (147456L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (147456L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 1024L)) % static_cast<long>(144L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr2[static_cast<long>((256L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L)))) % static_cast<long>(4L))) + (1024L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (1024L*x3) + (147456L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 147456L)) % static_cast<long>(8L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(2L))) + (4L*(c10::div_floor_integer((x2 + x2_inner), 2L))) + (64L*(c10::div_floor_integer((x2 + x2_inner + (32L*x1)), 64L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(2L))), 4L)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp5 = static_cast<float>(144.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = tmp9 - tmp12;
                            auto tmp14 = tmp1 * tmp13;
                            auto tmp15 = tmp0 + tmp14;
                            tmp15.store(out_ptr5 + static_cast<long>(x2 + (32L*x1) + (1024L*x3) + (147456L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_70 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(3.0517578125e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp4 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(7.62939453125e-06);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(7.62939453125e-06);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(7.62939453125e-06);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(7.62939453125e-06);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(7.62939453125e-06);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_89, primals_95, primals_101, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_152, primals_158, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_211, primals_212, primals_213, primals_312, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, mul_38, convolution_5, squeeze_16, mul_46, convolution_6, squeeze_19, add_34, convolution_7, squeeze_22, mul_61, convolution_8, squeeze_25, mul_69, convolution_9, squeeze_28, add_50, convolution_10, squeeze_31, mul_84, convolution_11, squeeze_34, mul_92, convolution_12, squeeze_37, add_66, convolution_13, squeeze_40, mul_107, convolution_14, squeeze_43, mul_115, convolution_15, squeeze_46, add_81, convolution_16, squeeze_49, mul_130, mul_131, view_3, getitem_36, getitem_37, getitem_38, getitem_40, getitem_41, getitem_42, getitem_45, getitem_46, view_7, mul_133, view_9, addmm_2, view_11, mul_136, view_13, getitem_52, getitem_53, getitem_54, getitem_56, getitem_57, getitem_58, getitem_61, getitem_62, view_17, mul_138, view_19, addmm_6, view_21, mul_141, view_25, convolution_18, squeeze_52, cat, convolution_19, squeeze_55, mul_158, convolution_20, squeeze_58, mul_166, convolution_21, squeeze_61, mul_174, convolution_22, squeeze_64, add_125, convolution_23, squeeze_67, mul_189, mul_190, view_29, getitem_82, getitem_83, getitem_84, getitem_86, getitem_87, getitem_88, getitem_91, getitem_92, view_33, mul_192, view_35, addmm_10, view_37, mul_195, view_39, getitem_98, getitem_99, getitem_100, getitem_102, getitem_103, getitem_104, getitem_107, getitem_108, view_43, mul_197, view_45, addmm_14, view_47, mul_200, view_49, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, getitem_123, getitem_124, view_53, mul_202, view_55, addmm_18, view_57, mul_205, view_59, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, getitem_139, getitem_140, view_63, mul_207, view_65, addmm_22, view_67, mul_210, view_71, convolution_25, squeeze_70, cat_1, convolution_26, squeeze_73, mul_227, convolution_27, squeeze_76, mul_235, convolution_28, squeeze_79, mul_243, convolution_29, squeeze_82, add_181, convolution_30, squeeze_85, mul_258, mul_259, view_75, getitem_160, getitem_161, getitem_162, getitem_164, getitem_165, getitem_166, getitem_169, getitem_170, view_79, mul_261, view_81, addmm_26, view_83, mul_264, view_85, getitem_176, getitem_177, getitem_178, getitem_180, getitem_181, getitem_182, getitem_185, getitem_186, view_89, mul_266, view_91, addmm_30, view_93, mul_269, view_95, getitem_192, getitem_193, getitem_194, getitem_196, getitem_197, getitem_198, getitem_201, getitem_202, view_99, mul_271, view_101, addmm_34, view_103, mul_274, view_107, convolution_32, squeeze_88, cat_2, convolution_33, squeeze_91, mul_291, convolution_34, squeeze_94, clone_64, permute_67, mul_301, unsqueeze_130, mul_313, unsqueeze_142, mul_325, unsqueeze_154, div_1, permute_76, permute_81, div_2, permute_85, alias_9, permute_91, div_3, permute_95, permute_100, div_4, permute_104, alias_10, permute_110, div_5, permute_114, permute_119, div_6, permute_123, alias_11, permute_129, div_7, mul_395, unsqueeze_166, unsqueeze_178, mul_416, unsqueeze_190, mul_428, unsqueeze_202, mul_440, unsqueeze_214, mul_452, unsqueeze_226, div_8, permute_142, permute_147, div_9, permute_151, alias_12, permute_157, div_10, permute_161, permute_166, div_11, permute_170, alias_13, permute_176, div_12, permute_180, permute_185, div_13, permute_189, alias_14, permute_195, div_14, permute_199, permute_204, div_15, permute_208, alias_15, permute_214, div_16, mul_539, unsqueeze_238, unsqueeze_250, mul_560, unsqueeze_262, mul_572, unsqueeze_274, mul_584, unsqueeze_286, mul_596, unsqueeze_298, div_17, permute_227, permute_232, div_18, permute_236, alias_16, permute_242, div_19, permute_246, permute_251, div_20, permute_255, alias_17, permute_261, div_21, mul_649, unsqueeze_310, unsqueeze_322, mul_670, unsqueeze_334, mul_682, unsqueeze_346, unsqueeze_358, mul_703, unsqueeze_370, mul_715, unsqueeze_382, unsqueeze_394, mul_736, unsqueeze_406, mul_748, unsqueeze_418, unsqueeze_430, mul_769, unsqueeze_442, mul_781, unsqueeze_454, unsqueeze_466, mul_802, unsqueeze_478, mul_814, unsqueeze_490, mul_826, unsqueeze_502, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_37, (96, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (160, ), (1, ))
    assert_size_stride(primals_57, (160, ), (1, ))
    assert_size_stride(primals_59, (160, ), (1, ))
    assert_size_stride(primals_61, (160, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_65, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_66, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_67, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_69, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_70, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_71, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_72, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_81, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_82, (144, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_83, (144, ), (1, ))
    assert_size_stride(primals_89, (144, ), (1, ))
    assert_size_stride(primals_95, (144, ), (1, ))
    assert_size_stride(primals_101, (144, ), (1, ))
    assert_size_stride(primals_107, (144, ), (1, ))
    assert_size_stride(primals_109, (96, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_110, (96, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_111, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_112, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_114, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_115, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_122, (192, ), (1, ))
    assert_size_stride(primals_128, (192, ), (1, ))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_166, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_167, (128, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_168, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (160, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_171, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_172, (240, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (240, ), (1, ))
    assert_size_stride(primals_179, (240, ), (1, ))
    assert_size_stride(primals_185, (240, ), (1, ))
    assert_size_stride(primals_191, (240, ), (1, ))
    assert_size_stride(primals_197, (240, ), (1, ))
    assert_size_stride(primals_203, (240, ), (1, ))
    assert_size_stride(primals_209, (240, ), (1, ))
    assert_size_stride(primals_211, (160, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_212, (160, 320, 3, 3), (2880, 1, 960, 320))
    assert_size_stride(primals_213, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_312, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 16, 128, 128), (262144, 1, 2048, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(mul_7, (8, 16, 128, 128), (262144, 1, 2048, 16))
    assert_size_stride(convolution_1, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(mul_15, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_2, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(mul_23, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_3, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_10, (32, ), (1, ))
    assert_size_stride(add_19, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_4, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(squeeze_13, (128, ), (1, ))
    assert_size_stride(mul_38, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(convolution_5, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_16, (128, ), (1, ))
    assert_size_stride(mul_46, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_6, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(add_34, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_7, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(mul_61, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_8, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_25, (256, ), (1, ))
    assert_size_stride(mul_69, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(add_50, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_10, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(mul_84, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_11, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(mul_92, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_12, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(add_66, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_13, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_40, (256, ), (1, ))
    assert_size_stride(mul_107, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_14, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_43, (256, ), (1, ))
    assert_size_stride(mul_115, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_15, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(squeeze_46, (96, ), (1, ))
    assert_size_stride(add_81, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(convolution_16, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(squeeze_49, (96, ), (1, ))
    assert_size_stride(mul_130, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(mul_131, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_3, (8192, 144), (144, 1))
    assert_size_stride(getitem_36, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_37, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_38, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_40, (32, 4, 256), (1024, 1, 4))
    assert_size_stride(getitem_41, (), ())
    assert_size_stride(getitem_42, (), ())
    assert_size_stride(getitem_45, (), ())
    assert_size_stride(getitem_46, (), ())
    assert_size_stride(view_7, (8192, 144), (144, 1))
    assert_size_stride(mul_133, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_9, (8192, 144), (144, 1))
    assert_size_stride(addmm_2, (8192, 288), (288, 1))
    assert_size_stride(view_11, (8192, 288), (288, 1))
    assert_size_stride(mul_136, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_13, (8192, 144), (144, 1))
    assert_size_stride(getitem_52, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_53, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_54, (32, 4, 256, 36), (110592, 36, 432, 1))
    assert_size_stride(getitem_56, (32, 4, 256), (1024, 1, 4))
    assert_size_stride(getitem_57, (), ())
    assert_size_stride(getitem_58, (), ())
    assert_size_stride(getitem_61, (), ())
    assert_size_stride(getitem_62, (), ())
    assert_size_stride(view_17, (8192, 144), (144, 1))
    assert_size_stride(mul_138, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_19, (8192, 144), (144, 1))
    assert_size_stride(addmm_6, (8192, 288), (288, 1))
    assert_size_stride(view_21, (8192, 288), (288, 1))
    assert_size_stride(mul_141, (32, 256, 144), (36864, 144, 1))
    assert_size_stride(view_25, (8, 144, 32, 32), (147456, 1, 4608, 144))
    assert_size_stride(convolution_18, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(squeeze_52, (96, ), (1, ))
    assert_size_stride(cat, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_19, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(squeeze_55, (96, ), (1, ))
    assert_size_stride(mul_158, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(convolution_20, (8, 384, 32, 32), (393216, 1, 12288, 384))
    assert_size_stride(squeeze_58, (384, ), (1, ))
    assert_size_stride(mul_166, (8, 384, 32, 32), (393216, 1, 12288, 384))
    assert_size_stride(convolution_21, (8, 384, 16, 16), (98304, 1, 6144, 384))
    assert_size_stride(squeeze_61, (384, ), (1, ))
    assert_size_stride(mul_174, (8, 384, 16, 16), (98304, 1, 6144, 384))
    assert_size_stride(convolution_22, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(add_125, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_23, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(mul_189, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(mul_190, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_29, (2048, 192), (192, 1))
    assert_size_stride(getitem_82, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_83, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_84, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_86, (32, 4, 64), (256, 1, 4))
    assert_size_stride(getitem_87, (), ())
    assert_size_stride(getitem_88, (), ())
    assert_size_stride(getitem_91, (), ())
    assert_size_stride(getitem_92, (), ())
    assert_size_stride(view_33, (2048, 192), (192, 1))
    assert_size_stride(mul_192, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_35, (2048, 192), (192, 1))
    assert_size_stride(addmm_10, (2048, 384), (384, 1))
    assert_size_stride(view_37, (2048, 384), (384, 1))
    assert_size_stride(mul_195, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_39, (2048, 192), (192, 1))
    assert_size_stride(getitem_98, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_99, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_100, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_102, (32, 4, 64), (256, 1, 4))
    assert_size_stride(getitem_103, (), ())
    assert_size_stride(getitem_104, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(getitem_108, (), ())
    assert_size_stride(view_43, (2048, 192), (192, 1))
    assert_size_stride(mul_197, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_45, (2048, 192), (192, 1))
    assert_size_stride(addmm_14, (2048, 384), (384, 1))
    assert_size_stride(view_47, (2048, 384), (384, 1))
    assert_size_stride(mul_200, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_49, (2048, 192), (192, 1))
    assert_size_stride(getitem_114, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_115, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_116, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_118, (32, 4, 64), (256, 1, 4))
    assert_size_stride(getitem_119, (), ())
    assert_size_stride(getitem_120, (), ())
    assert_size_stride(getitem_123, (), ())
    assert_size_stride(getitem_124, (), ())
    assert_size_stride(view_53, (2048, 192), (192, 1))
    assert_size_stride(mul_202, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_55, (2048, 192), (192, 1))
    assert_size_stride(addmm_18, (2048, 384), (384, 1))
    assert_size_stride(view_57, (2048, 384), (384, 1))
    assert_size_stride(mul_205, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_59, (2048, 192), (192, 1))
    assert_size_stride(getitem_130, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_131, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_132, (32, 4, 64, 48), (36864, 48, 576, 1))
    assert_size_stride(getitem_134, (32, 4, 64), (256, 1, 4))
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(getitem_139, (), ())
    assert_size_stride(getitem_140, (), ())
    assert_size_stride(view_63, (2048, 192), (192, 1))
    assert_size_stride(mul_207, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_65, (2048, 192), (192, 1))
    assert_size_stride(addmm_22, (2048, 384), (384, 1))
    assert_size_stride(view_67, (2048, 384), (384, 1))
    assert_size_stride(mul_210, (32, 64, 192), (12288, 192, 1))
    assert_size_stride(view_71, (8, 192, 16, 16), (49152, 1, 3072, 192))
    assert_size_stride(convolution_25, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(cat_1, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_26, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(squeeze_73, (128, ), (1, ))
    assert_size_stride(mul_227, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_27, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(mul_235, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_28, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_79, (512, ), (1, ))
    assert_size_stride(mul_243, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_29, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(squeeze_82, (160, ), (1, ))
    assert_size_stride(add_181, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(convolution_30, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(squeeze_85, (160, ), (1, ))
    assert_size_stride(mul_258, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(mul_259, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_75, (512, 240), (240, 1))
    assert_size_stride(getitem_160, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_161, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_162, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_164, (32, 4, 16), (64, 1, 4))
    assert_size_stride(getitem_165, (), ())
    assert_size_stride(getitem_166, (), ())
    assert_size_stride(getitem_169, (), ())
    assert_size_stride(getitem_170, (), ())
    assert_size_stride(view_79, (512, 240), (240, 1))
    assert_size_stride(mul_261, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_81, (512, 240), (240, 1))
    assert_size_stride(addmm_26, (512, 480), (480, 1))
    assert_size_stride(view_83, (512, 480), (480, 1))
    assert_size_stride(mul_264, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_85, (512, 240), (240, 1))
    assert_size_stride(getitem_176, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_177, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_178, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_180, (32, 4, 16), (64, 1, 4))
    assert_size_stride(getitem_181, (), ())
    assert_size_stride(getitem_182, (), ())
    assert_size_stride(getitem_185, (), ())
    assert_size_stride(getitem_186, (), ())
    assert_size_stride(view_89, (512, 240), (240, 1))
    assert_size_stride(mul_266, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_91, (512, 240), (240, 1))
    assert_size_stride(addmm_30, (512, 480), (480, 1))
    assert_size_stride(view_93, (512, 480), (480, 1))
    assert_size_stride(mul_269, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_95, (512, 240), (240, 1))
    assert_size_stride(getitem_192, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_193, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_194, (32, 4, 16, 60), (11520, 60, 720, 1))
    assert_size_stride(getitem_196, (32, 4, 16), (64, 1, 4))
    assert_size_stride(getitem_197, (), ())
    assert_size_stride(getitem_198, (), ())
    assert_size_stride(getitem_201, (), ())
    assert_size_stride(getitem_202, (), ())
    assert_size_stride(view_99, (512, 240), (240, 1))
    assert_size_stride(mul_271, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_101, (512, 240), (240, 1))
    assert_size_stride(addmm_34, (512, 480), (480, 1))
    assert_size_stride(view_103, (512, 480), (480, 1))
    assert_size_stride(mul_274, (32, 16, 240), (3840, 240, 1))
    assert_size_stride(view_107, (8, 240, 8, 8), (15360, 1, 1920, 240))
    assert_size_stride(convolution_32, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(squeeze_88, (160, ), (1, ))
    assert_size_stride(cat_2, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(convolution_33, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(squeeze_91, (160, ), (1, ))
    assert_size_stride(mul_291, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(convolution_34, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_94, (640, ), (1, ))
    assert_size_stride(clone_64, (8, 640), (640, 1))
    assert_size_stride(permute_67, (1000, 640), (640, 1))
    assert_size_stride(mul_301, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(unsqueeze_130, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(mul_313, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(unsqueeze_142, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(mul_325, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(unsqueeze_154, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(div_1, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_76, (240, 480), (480, 1))
    assert_size_stride(permute_81, (480, 240), (240, 1))
    assert_size_stride(div_2, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_85, (240, 240), (240, 1))
    assert_size_stride(alias_9, (32, 4, 16, 60), (3840, 1, 240, 4))
    assert_size_stride(permute_91, (720, 240), (240, 1))
    assert_size_stride(div_3, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_95, (240, 480), (480, 1))
    assert_size_stride(permute_100, (480, 240), (240, 1))
    assert_size_stride(div_4, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_104, (240, 240), (240, 1))
    assert_size_stride(alias_10, (32, 4, 16, 60), (3840, 1, 240, 4))
    assert_size_stride(permute_110, (720, 240), (240, 1))
    assert_size_stride(div_5, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_114, (240, 480), (480, 1))
    assert_size_stride(permute_119, (480, 240), (240, 1))
    assert_size_stride(div_6, (32, 16, 1), (16, 1, 1))
    assert_size_stride(permute_123, (240, 240), (240, 1))
    assert_size_stride(alias_11, (32, 4, 16, 60), (3840, 1, 240, 4))
    assert_size_stride(permute_129, (720, 240), (240, 1))
    assert_size_stride(div_7, (32, 16, 1), (16, 1, 1))
    assert_size_stride(mul_395, (8, 160, 8, 8), (10240, 1, 1280, 160))
    assert_size_stride(unsqueeze_166, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(mul_416, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_190, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_428, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(unsqueeze_202, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_440, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(unsqueeze_214, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_452, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(unsqueeze_226, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(div_8, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_142, (192, 384), (384, 1))
    assert_size_stride(permute_147, (384, 192), (192, 1))
    assert_size_stride(div_9, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_151, (192, 192), (192, 1))
    assert_size_stride(alias_12, (32, 4, 64, 48), (12288, 1, 192, 4))
    assert_size_stride(permute_157, (576, 192), (192, 1))
    assert_size_stride(div_10, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_161, (192, 384), (384, 1))
    assert_size_stride(permute_166, (384, 192), (192, 1))
    assert_size_stride(div_11, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_170, (192, 192), (192, 1))
    assert_size_stride(alias_13, (32, 4, 64, 48), (12288, 1, 192, 4))
    assert_size_stride(permute_176, (576, 192), (192, 1))
    assert_size_stride(div_12, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_180, (192, 384), (384, 1))
    assert_size_stride(permute_185, (384, 192), (192, 1))
    assert_size_stride(div_13, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_189, (192, 192), (192, 1))
    assert_size_stride(alias_14, (32, 4, 64, 48), (12288, 1, 192, 4))
    assert_size_stride(permute_195, (576, 192), (192, 1))
    assert_size_stride(div_14, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_199, (192, 384), (384, 1))
    assert_size_stride(permute_204, (384, 192), (192, 1))
    assert_size_stride(div_15, (32, 64, 1), (64, 1, 1))
    assert_size_stride(permute_208, (192, 192), (192, 1))
    assert_size_stride(alias_15, (32, 4, 64, 48), (12288, 1, 192, 4))
    assert_size_stride(permute_214, (576, 192), (192, 1))
    assert_size_stride(div_16, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_539, (8, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(unsqueeze_238, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_560, (8, 384, 16, 16), (98304, 1, 6144, 384))
    assert_size_stride(unsqueeze_262, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(mul_572, (8, 384, 32, 32), (393216, 1, 12288, 384))
    assert_size_stride(unsqueeze_274, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(mul_584, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(unsqueeze_286, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_596, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(unsqueeze_298, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(div_17, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_227, (144, 288), (288, 1))
    assert_size_stride(permute_232, (288, 144), (144, 1))
    assert_size_stride(div_18, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_236, (144, 144), (144, 1))
    assert_size_stride(alias_16, (32, 4, 256, 36), (36864, 1, 144, 4))
    assert_size_stride(permute_242, (432, 144), (144, 1))
    assert_size_stride(div_19, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_246, (144, 288), (288, 1))
    assert_size_stride(permute_251, (288, 144), (144, 1))
    assert_size_stride(div_20, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_255, (144, 144), (144, 1))
    assert_size_stride(alias_17, (32, 4, 256, 36), (36864, 1, 144, 4))
    assert_size_stride(permute_261, (432, 144), (144, 1))
    assert_size_stride(div_21, (32, 256, 1), (256, 1, 1))
    assert_size_stride(mul_649, (8, 96, 32, 32), (98304, 1, 3072, 96))
    assert_size_stride(unsqueeze_310, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_670, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(unsqueeze_334, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_682, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_346, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_703, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_370, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_715, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_382, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_736, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_406, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_748, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_418, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_769, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(unsqueeze_442, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_781, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(unsqueeze_454, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_802, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(unsqueeze_478, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_814, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(unsqueeze_490, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_826, (8, 16, 128, 128), (262144, 1, 2048, 16))
    assert_size_stride(unsqueeze_502, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_67, out=buf0)
    del permute_67
    buf1 = empty((1000, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_64, out=buf1)
    del clone_64
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((640, ), device='cpu', dtype=torch.float32)
    buf4 = empty((640, ), device='cpu', dtype=torch.float32)
    buf5 = empty((640, ), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 640, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_301.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_130.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del buf4
    del convolution_34
    del mul_301
    del primals_63
    del squeeze_94
    del tangents_1
    del unsqueeze_130
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, mul_291, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del mul_291
    del primals_213
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((160, ), device='cpu', dtype=torch.float32)
    buf11 = empty((160, ), device='cpu', dtype=torch.float32)
    buf12 = empty((160, ), device='cpu', dtype=torch.float32)
    buf13 = buf8; del buf8  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_1(c_void_p(buf13.data_ptr()), c_void_p(mul_313.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_142.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del convolution_33
    del mul_313
    del primals_61
    del squeeze_91
    del unsqueeze_142
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, cat_2, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_2
    del primals_212
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = buf11; del buf11  # reuse
    buf18 = empty((160, ), device='cpu', dtype=torch.float32)
    buf19 = empty((160, ), device='cpu', dtype=torch.float32)
    buf20 = buf13; del buf13  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_2(c_void_p(buf15.data_ptr()), c_void_p(mul_325.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_154.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    del convolution_32
    del mul_325
    del primals_59
    del squeeze_88
    del unsqueeze_154
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf21 = aten.convolution_backward(buf20, view_107, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf20
    del primals_211
    del view_107
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = empty_strided((32, 16, 1), (1, 32, 512), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((32, 16, 1), (16, 1, 512), device='cpu', dtype=torch.float32)
    buf26 = empty((32, 16, 240), device='cpu', dtype=torch.float32)
    buf27 = empty((240, ), device='cpu', dtype=torch.float32)
    buf28 = empty((240, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_3(c_void_p(buf22.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(mul_274.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del div_1
    del mul_274
    del primals_209
    buf29 = empty((512, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (512, 240), (240, 1), 0), permute_76, out=buf29)
    del permute_76
    buf30 = empty((240, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (240, 512), (1, 240), 0), view_103, out=buf30)
    del view_103
    buf31 = empty((1, 240), device='cpu', dtype=torch.float32)
    buf32 = reinterpret_tensor(buf29, (32, 16, 480), (7680, 480, 1), 0); del buf29  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_4(c_void_p(buf32.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf31.data_ptr()))
    del addmm_34
    buf33 = reinterpret_tensor(buf22, (512, 240), (240, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (512, 480), (480, 1), 0), permute_81, out=buf33)
    del permute_81
    buf34 = empty((480, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (480, 512), (1, 480), 0), view_101, out=buf34)
    del view_101
    buf35 = empty((1, 480), device='cpu', dtype=torch.float32)
    buf36 = buf25; del buf25  # reuse
    buf37 = reinterpret_tensor(buf24, (32, 16, 1), (16, 1, 512), 0); del buf24  # reuse
    buf38 = empty((240, ), device='cpu', dtype=torch.float32)
    buf39 = empty((240, ), device='cpu', dtype=torch.float32)
    buf40 = buf26; del buf26  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_5(c_void_p(buf40.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(mul_271.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    del div_2
    del mul_271
    del primals_203
    buf41 = buf33; del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (512, 240), (240, 1), 0), permute_85, out=buf41)
    del permute_85
    buf42 = empty((240, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (240, 512), (1, 240), 0), view_99, out=buf42)
    del view_99
    buf43 = empty((1, 240), device='cpu', dtype=torch.float32)
    cpp_fused_sum_6(c_void_p(buf40.data_ptr()), c_void_p(buf43.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf44 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf41, (32, 4, 16, 60), (3840, 60, 240, 1), 0), getitem_192, getitem_193, getitem_194, alias_9, getitem_196, getitem_197, getitem_198, 0, 0, 0.0, False, getitem_201, getitem_202)
    del alias_9
    del buf41
    del getitem_192
    del getitem_193
    del getitem_194
    del getitem_196
    del getitem_197
    del getitem_198
    del getitem_201
    del getitem_202
    buf45 = buf44[0]
    buf46 = buf44[1]
    buf47 = buf44[2]
    del buf44
    buf48 = empty((32, 16, 3, 4, 60), device='cpu', dtype=torch.float32)
    cpp_fused_clone_7(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del buf45
    del buf46
    buf49 = reinterpret_tensor(buf47, (512, 240), (240, 1), 0); del buf47  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (512, 720), (720, 1), 0), permute_91, out=buf49)
    del permute_91
    buf50 = empty((720, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (720, 512), (1, 720), 0), view_95, out=buf50)
    del view_95
    buf51 = empty((1, 720), device='cpu', dtype=torch.float32)
    buf52 = buf37; del buf37  # reuse
    buf53 = buf36; del buf36  # reuse
    buf54 = empty((240, ), device='cpu', dtype=torch.float32)
    buf55 = empty((240, ), device='cpu', dtype=torch.float32)
    buf56 = buf40; del buf40  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_8(c_void_p(buf56.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(mul_269.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del div_3
    del mul_269
    del primals_197
    buf57 = reinterpret_tensor(buf32, (512, 480), (480, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf56, (512, 240), (240, 1), 0), permute_95, out=buf57)
    del permute_95
    buf58 = empty((240, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf56, (240, 512), (1, 240), 0), view_93, out=buf58)
    del view_93
    buf59 = empty((1, 240), device='cpu', dtype=torch.float32)
    buf60 = reinterpret_tensor(buf57, (32, 16, 480), (7680, 480, 1), 0); del buf57  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_9(c_void_p(buf60.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(buf59.data_ptr()))
    del addmm_30
    buf61 = buf49; del buf49  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (512, 480), (480, 1), 0), permute_100, out=buf61)
    del permute_100
    buf62 = empty((480, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (480, 512), (1, 480), 0), view_91, out=buf62)
    del view_91
    buf63 = empty((1, 480), device='cpu', dtype=torch.float32)
    buf64 = buf53; del buf53  # reuse
    buf65 = buf52; del buf52  # reuse
    buf66 = empty((240, ), device='cpu', dtype=torch.float32)
    buf67 = empty((240, ), device='cpu', dtype=torch.float32)
    buf68 = buf56; del buf56  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_10(c_void_p(buf68.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(mul_266.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del div_4
    del mul_266
    del primals_191
    buf69 = buf61; del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (512, 240), (240, 1), 0), permute_104, out=buf69)
    del permute_104
    buf70 = empty((240, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (240, 512), (1, 240), 0), view_89, out=buf70)
    del view_89
    buf71 = empty((1, 240), device='cpu', dtype=torch.float32)
    cpp_fused_sum_11(c_void_p(buf68.data_ptr()), c_void_p(buf71.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf72 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf69, (32, 4, 16, 60), (3840, 60, 240, 1), 0), getitem_176, getitem_177, getitem_178, alias_10, getitem_180, getitem_181, getitem_182, 0, 0, 0.0, False, getitem_185, getitem_186)
    del alias_10
    del buf69
    del getitem_176
    del getitem_177
    del getitem_178
    del getitem_180
    del getitem_181
    del getitem_182
    del getitem_185
    del getitem_186
    buf73 = buf72[0]
    buf74 = buf72[1]
    buf75 = buf72[2]
    del buf72
    buf76 = buf48; del buf48  # reuse
    cpp_fused_clone_12(c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del buf73
    del buf74
    buf77 = reinterpret_tensor(buf75, (512, 240), (240, 1), 0); del buf75  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (512, 720), (720, 1), 0), permute_110, out=buf77)
    del permute_110
    buf78 = empty((720, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (720, 512), (1, 720), 0), view_85, out=buf78)
    del view_85
    buf79 = empty((1, 720), device='cpu', dtype=torch.float32)
    buf80 = buf65; del buf65  # reuse
    buf81 = buf64; del buf64  # reuse
    buf82 = empty((240, ), device='cpu', dtype=torch.float32)
    buf83 = empty((240, ), device='cpu', dtype=torch.float32)
    buf84 = buf68; del buf68  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_13(c_void_p(buf84.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(mul_264.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del div_5
    del mul_264
    del primals_185
    buf85 = reinterpret_tensor(buf60, (512, 480), (480, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (512, 240), (240, 1), 0), permute_114, out=buf85)
    del permute_114
    buf86 = empty((240, 480), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (240, 512), (1, 240), 0), view_83, out=buf86)
    del view_83
    buf87 = empty((1, 240), device='cpu', dtype=torch.float32)
    buf88 = reinterpret_tensor(buf85, (32, 16, 480), (7680, 480, 1), 0); del buf85  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_14(c_void_p(buf88.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf87.data_ptr()))
    del addmm_26
    buf89 = buf77; del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf88, (512, 480), (480, 1), 0), permute_119, out=buf89)
    del permute_119
    buf90 = empty((480, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf88, (480, 512), (1, 480), 0), view_81, out=buf90)
    del view_81
    buf91 = empty((1, 480), device='cpu', dtype=torch.float32)
    buf92 = buf81; del buf81  # reuse
    buf93 = buf80; del buf80  # reuse
    buf94 = empty((240, ), device='cpu', dtype=torch.float32)
    buf95 = empty((240, ), device='cpu', dtype=torch.float32)
    buf96 = buf84; del buf84  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_15(c_void_p(buf96.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(mul_261.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del buf88
    del div_6
    del mul_261
    del primals_179
    buf97 = buf89; del buf89  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (512, 240), (240, 1), 0), permute_123, out=buf97)
    del permute_123
    buf98 = empty((240, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (240, 512), (1, 240), 0), view_79, out=buf98)
    del view_79
    buf99 = empty((1, 240), device='cpu', dtype=torch.float32)
    cpp_fused_sum_16(c_void_p(buf96.data_ptr()), c_void_p(buf99.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf100 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf97, (32, 4, 16, 60), (3840, 60, 240, 1), 0), getitem_160, getitem_161, getitem_162, alias_11, getitem_164, getitem_165, getitem_166, 0, 0, 0.0, False, getitem_169, getitem_170)
    del alias_11
    del buf97
    del getitem_160
    del getitem_161
    del getitem_162
    del getitem_164
    del getitem_165
    del getitem_166
    del getitem_169
    del getitem_170
    buf101 = buf100[0]
    buf102 = buf100[1]
    buf103 = buf100[2]
    del buf100
    buf104 = buf76; del buf76  # reuse
    cpp_fused_clone_17(c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del buf101
    buf105 = reinterpret_tensor(buf103, (512, 240), (240, 1), 0); del buf103  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (512, 720), (720, 1), 0), permute_129, out=buf105)
    del permute_129
    buf106 = empty((720, 240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (720, 512), (1, 720), 0), view_75, out=buf106)
    del view_75
    buf107 = empty((1, 720), device='cpu', dtype=torch.float32)
    buf108 = buf93; del buf93  # reuse
    buf109 = buf92; del buf92  # reuse
    buf110 = empty((240, ), device='cpu', dtype=torch.float32)
    buf111 = empty((240, ), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf102, (8, 240, 8, 8), (15360, 64, 8, 1), 0); del buf102  # reuse
    cpp_fused_convolution_backward_native_layer_norm_backward_sum_18(c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(mul_259.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del buf104
    del buf105
    del buf96
    del div_7
    del mul_259
    del primals_173
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf113 = aten.convolution_backward(buf112, mul_258, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf112
    del mul_258
    del primals_172
    buf114 = buf113[0]
    buf115 = buf113[1]
    del buf113
    buf116 = buf18; del buf18  # reuse
    buf117 = empty((160, ), device='cpu', dtype=torch.float32)
    buf118 = empty((160, ), device='cpu', dtype=torch.float32)
    buf119 = buf114; del buf114  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_19(c_void_p(buf119.data_ptr()), c_void_p(mul_395.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_166.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    del convolution_30
    del mul_395
    del primals_57
    del squeeze_85
    del unsqueeze_166
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf120 = aten.convolution_backward(buf119, add_181, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_181
    del buf119
    del primals_171
    buf121 = buf120[0]
    buf122 = buf120[1]
    del buf120
    buf123 = buf117; del buf117  # reuse
    buf124 = empty((160, ), device='cpu', dtype=torch.float32)
    buf125 = empty((160, ), device='cpu', dtype=torch.float32)
    buf126 = buf121; del buf121  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_20(c_void_p(buf126.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_178.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    del buf124
    del buf15
    del convolution_29
    del primals_55
    del squeeze_82
    del unsqueeze_178
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf127 = aten.convolution_backward(buf126, mul_243, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf126
    del mul_243
    del primals_170
    buf128 = buf127[0]
    buf129 = buf127[1]
    del buf127
    buf130 = reinterpret_tensor(buf109, (512, ), (1, ), 0); del buf109  # reuse
    buf131 = reinterpret_tensor(buf108, (512, ), (1, ), 0); del buf108  # reuse
    buf132 = empty((512, ), device='cpu', dtype=torch.float32)
    buf133 = buf128; del buf128  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_21(c_void_p(buf133.data_ptr()), c_void_p(mul_416.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_190.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del convolution_28
    del mul_416
    del primals_53
    del squeeze_79
    del unsqueeze_190
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf134 = aten.convolution_backward(buf133, mul_235, primals_169, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False])
    del buf133
    del mul_235
    del primals_169
    buf135 = buf134[0]
    buf136 = buf134[1]
    del buf134
    buf137 = buf131; del buf131  # reuse
    buf138 = empty((512, ), device='cpu', dtype=torch.float32)
    buf139 = empty((512, ), device='cpu', dtype=torch.float32)
    buf140 = buf135; del buf135  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_22(c_void_p(buf140.data_ptr()), c_void_p(mul_428.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_202.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del buf138
    del convolution_27
    del mul_428
    del primals_51
    del squeeze_76
    del unsqueeze_202
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf141 = aten.convolution_backward(buf140, mul_227, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf140
    del mul_227
    del primals_168
    buf142 = buf141[0]
    buf143 = buf141[1]
    del buf141
    buf144 = empty((128, ), device='cpu', dtype=torch.float32)
    buf145 = empty((128, ), device='cpu', dtype=torch.float32)
    buf146 = empty((128, ), device='cpu', dtype=torch.float32)
    buf147 = buf142; del buf142  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_23(c_void_p(buf147.data_ptr()), c_void_p(mul_440.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_214.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del convolution_26
    del mul_440
    del primals_49
    del squeeze_73
    del unsqueeze_214
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf148 = aten.convolution_backward(buf147, cat_1, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_1
    del primals_167
    buf149 = buf148[0]
    buf150 = buf148[1]
    del buf148
    buf151 = buf145; del buf145  # reuse
    buf152 = empty((128, ), device='cpu', dtype=torch.float32)
    buf153 = empty((128, ), device='cpu', dtype=torch.float32)
    buf154 = buf147; del buf147  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_24(c_void_p(buf149.data_ptr()), c_void_p(mul_452.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_226.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del convolution_25
    del mul_452
    del primals_47
    del squeeze_70
    del unsqueeze_226
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf155 = aten.convolution_backward(buf154, view_71, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf154
    del primals_166
    del view_71
    buf156 = buf155[0]
    buf157 = buf155[1]
    del buf155
    buf158 = empty_strided((32, 64, 1), (1, 32, 2048), device='cpu', dtype=torch.float32)
    buf159 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf160 = empty((32, 64, 192), device='cpu', dtype=torch.float32)
    buf161 = empty((192, ), device='cpu', dtype=torch.float32)
    buf162 = empty((192, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_25(c_void_p(buf156.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(mul_210.data_ptr()), c_void_p(div_8.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()))
    del div_8
    del mul_210
    del primals_164
    buf163 = empty((2048, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf160, (2048, 192), (192, 1), 0), permute_142, out=buf163)
    del permute_142
    buf164 = empty((192, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf160, (192, 2048), (1, 192), 0), view_67, out=buf164)
    del view_67
    buf165 = empty((1, 192), device='cpu', dtype=torch.float32)
    buf166 = reinterpret_tensor(buf163, (32, 64, 384), (24576, 384, 1), 0); del buf163  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_26(c_void_p(buf166.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf165.data_ptr()))
    del addmm_22
    buf167 = reinterpret_tensor(buf156, (2048, 192), (192, 1), 0); del buf156  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (2048, 384), (384, 1), 0), permute_147, out=buf167)
    del permute_147
    buf168 = empty((384, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (384, 2048), (1, 384), 0), view_65, out=buf168)
    del view_65
    buf169 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf170 = buf159; del buf159  # reuse
    buf171 = reinterpret_tensor(buf158, (32, 64, 1), (64, 1, 2048), 0); del buf158  # reuse
    buf172 = empty((192, ), device='cpu', dtype=torch.float32)
    buf173 = empty((192, ), device='cpu', dtype=torch.float32)
    buf174 = buf160; del buf160  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf174.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(mul_207.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del div_9
    del mul_207
    del primals_158
    buf175 = buf167; del buf167  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (2048, 192), (192, 1), 0), permute_151, out=buf175)
    del permute_151
    buf176 = empty((192, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (192, 2048), (1, 192), 0), view_63, out=buf176)
    del view_63
    buf177 = empty((1, 192), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf174.data_ptr()), c_void_p(buf177.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf178 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf175, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_130, getitem_131, getitem_132, alias_12, getitem_134, getitem_135, getitem_136, 0, 0, 0.0, False, getitem_139, getitem_140)
    del alias_12
    del buf175
    del getitem_130
    del getitem_131
    del getitem_132
    del getitem_134
    del getitem_135
    del getitem_136
    del getitem_139
    del getitem_140
    buf179 = buf178[0]
    buf180 = buf178[1]
    buf181 = buf178[2]
    del buf178
    buf182 = empty((32, 64, 3, 4, 48), device='cpu', dtype=torch.float32)
    cpp_fused_clone_29(c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    del buf179
    del buf180
    buf183 = reinterpret_tensor(buf181, (2048, 192), (192, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (2048, 576), (576, 1), 0), permute_157, out=buf183)
    del permute_157
    buf184 = empty((576, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (576, 2048), (1, 576), 0), view_59, out=buf184)
    del view_59
    buf185 = empty((1, 576), device='cpu', dtype=torch.float32)
    buf186 = buf171; del buf171  # reuse
    buf187 = buf170; del buf170  # reuse
    buf188 = empty((192, ), device='cpu', dtype=torch.float32)
    buf189 = empty((192, ), device='cpu', dtype=torch.float32)
    buf190 = buf174; del buf174  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_30(c_void_p(buf190.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(mul_205.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    del div_10
    del mul_205
    del primals_152
    buf191 = reinterpret_tensor(buf166, (2048, 384), (384, 1), 0); del buf166  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (2048, 192), (192, 1), 0), permute_161, out=buf191)
    del permute_161
    buf192 = empty((192, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (192, 2048), (1, 192), 0), view_57, out=buf192)
    del view_57
    buf193 = empty((1, 192), device='cpu', dtype=torch.float32)
    buf194 = reinterpret_tensor(buf191, (32, 64, 384), (24576, 384, 1), 0); del buf191  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_31(c_void_p(buf194.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf193.data_ptr()))
    del addmm_18
    buf195 = buf183; del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (2048, 384), (384, 1), 0), permute_166, out=buf195)
    del permute_166
    buf196 = empty((384, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (384, 2048), (1, 384), 0), view_55, out=buf196)
    del view_55
    buf197 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf198 = buf187; del buf187  # reuse
    buf199 = buf186; del buf186  # reuse
    buf200 = empty((192, ), device='cpu', dtype=torch.float32)
    buf201 = empty((192, ), device='cpu', dtype=torch.float32)
    buf202 = buf190; del buf190  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_32(c_void_p(buf202.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(mul_202.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del div_11
    del mul_202
    del primals_146
    buf203 = buf195; del buf195  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (2048, 192), (192, 1), 0), permute_170, out=buf203)
    del permute_170
    buf204 = empty((192, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (192, 2048), (1, 192), 0), view_53, out=buf204)
    del view_53
    buf205 = empty((1, 192), device='cpu', dtype=torch.float32)
    cpp_fused_sum_33(c_void_p(buf202.data_ptr()), c_void_p(buf205.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf206 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf203, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_114, getitem_115, getitem_116, alias_13, getitem_118, getitem_119, getitem_120, 0, 0, 0.0, False, getitem_123, getitem_124)
    del alias_13
    del buf203
    del getitem_114
    del getitem_115
    del getitem_116
    del getitem_118
    del getitem_119
    del getitem_120
    del getitem_123
    del getitem_124
    buf207 = buf206[0]
    buf208 = buf206[1]
    buf209 = buf206[2]
    del buf206
    buf210 = buf182; del buf182  # reuse
    cpp_fused_clone_34(c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    del buf207
    del buf208
    buf211 = reinterpret_tensor(buf209, (2048, 192), (192, 1), 0); del buf209  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (2048, 576), (576, 1), 0), permute_176, out=buf211)
    del permute_176
    buf212 = empty((576, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (576, 2048), (1, 576), 0), view_49, out=buf212)
    del view_49
    buf213 = empty((1, 576), device='cpu', dtype=torch.float32)
    buf214 = buf199; del buf199  # reuse
    buf215 = buf198; del buf198  # reuse
    buf216 = empty((192, ), device='cpu', dtype=torch.float32)
    buf217 = empty((192, ), device='cpu', dtype=torch.float32)
    buf218 = buf202; del buf202  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_35(c_void_p(buf218.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(mul_200.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    del div_12
    del mul_200
    del primals_140
    buf219 = reinterpret_tensor(buf194, (2048, 384), (384, 1), 0); del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (2048, 192), (192, 1), 0), permute_180, out=buf219)
    del permute_180
    buf220 = empty((192, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (192, 2048), (1, 192), 0), view_47, out=buf220)
    del view_47
    buf221 = empty((1, 192), device='cpu', dtype=torch.float32)
    buf222 = reinterpret_tensor(buf219, (32, 64, 384), (24576, 384, 1), 0); del buf219  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_36(c_void_p(buf222.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf221.data_ptr()))
    del addmm_14
    buf223 = buf211; del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (2048, 384), (384, 1), 0), permute_185, out=buf223)
    del permute_185
    buf224 = empty((384, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (384, 2048), (1, 384), 0), view_45, out=buf224)
    del view_45
    buf225 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf226 = buf215; del buf215  # reuse
    buf227 = buf214; del buf214  # reuse
    buf228 = empty((192, ), device='cpu', dtype=torch.float32)
    buf229 = empty((192, ), device='cpu', dtype=torch.float32)
    buf230 = buf218; del buf218  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_37(c_void_p(buf230.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(mul_197.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    del div_13
    del mul_197
    del primals_134
    buf231 = buf223; del buf223  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (2048, 192), (192, 1), 0), permute_189, out=buf231)
    del permute_189
    buf232 = empty((192, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (192, 2048), (1, 192), 0), view_43, out=buf232)
    del view_43
    buf233 = empty((1, 192), device='cpu', dtype=torch.float32)
    cpp_fused_sum_38(c_void_p(buf230.data_ptr()), c_void_p(buf233.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf234 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf231, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_98, getitem_99, getitem_100, alias_14, getitem_102, getitem_103, getitem_104, 0, 0, 0.0, False, getitem_107, getitem_108)
    del alias_14
    del buf231
    del getitem_100
    del getitem_102
    del getitem_103
    del getitem_104
    del getitem_107
    del getitem_108
    del getitem_98
    del getitem_99
    buf235 = buf234[0]
    buf236 = buf234[1]
    buf237 = buf234[2]
    del buf234
    buf238 = buf210; del buf210  # reuse
    cpp_fused_clone_39(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del buf235
    del buf236
    buf239 = reinterpret_tensor(buf237, (2048, 192), (192, 1), 0); del buf237  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf238, (2048, 576), (576, 1), 0), permute_195, out=buf239)
    del permute_195
    buf240 = empty((576, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf238, (576, 2048), (1, 576), 0), view_39, out=buf240)
    del view_39
    buf241 = empty((1, 576), device='cpu', dtype=torch.float32)
    buf242 = buf227; del buf227  # reuse
    buf243 = buf226; del buf226  # reuse
    buf244 = empty((192, ), device='cpu', dtype=torch.float32)
    buf245 = empty((192, ), device='cpu', dtype=torch.float32)
    buf246 = buf230; del buf230  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_40(c_void_p(buf246.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(mul_195.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    del div_14
    del mul_195
    del primals_128
    buf247 = reinterpret_tensor(buf222, (2048, 384), (384, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf246, (2048, 192), (192, 1), 0), permute_199, out=buf247)
    del permute_199
    buf248 = empty((192, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf246, (192, 2048), (1, 192), 0), view_37, out=buf248)
    del view_37
    buf249 = empty((1, 192), device='cpu', dtype=torch.float32)
    buf250 = reinterpret_tensor(buf247, (32, 64, 384), (24576, 384, 1), 0); del buf247  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_41(c_void_p(buf250.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf249.data_ptr()))
    del addmm_10
    buf251 = buf239; del buf239  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf250, (2048, 384), (384, 1), 0), permute_204, out=buf251)
    del permute_204
    buf252 = empty((384, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf250, (384, 2048), (1, 384), 0), view_35, out=buf252)
    del view_35
    buf253 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf254 = buf243; del buf243  # reuse
    buf255 = buf242; del buf242  # reuse
    buf256 = empty((192, ), device='cpu', dtype=torch.float32)
    buf257 = empty((192, ), device='cpu', dtype=torch.float32)
    buf258 = buf246; del buf246  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_42(c_void_p(buf258.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(mul_192.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    del buf250
    del div_15
    del mul_192
    del primals_122
    buf259 = buf251; del buf251  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (2048, 192), (192, 1), 0), permute_208, out=buf259)
    del permute_208
    buf260 = empty((192, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (192, 2048), (1, 192), 0), view_33, out=buf260)
    del view_33
    buf261 = empty((1, 192), device='cpu', dtype=torch.float32)
    cpp_fused_sum_43(c_void_p(buf258.data_ptr()), c_void_p(buf261.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf262 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf259, (32, 4, 64, 48), (12288, 48, 192, 1), 0), getitem_82, getitem_83, getitem_84, alias_15, getitem_86, getitem_87, getitem_88, 0, 0, 0.0, False, getitem_91, getitem_92)
    del alias_15
    del buf259
    del getitem_82
    del getitem_83
    del getitem_84
    del getitem_86
    del getitem_87
    del getitem_88
    del getitem_91
    del getitem_92
    buf263 = buf262[0]
    buf264 = buf262[1]
    buf265 = buf262[2]
    del buf262
    buf266 = buf238; del buf238  # reuse
    cpp_fused_clone_44(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del buf263
    buf267 = reinterpret_tensor(buf265, (2048, 192), (192, 1), 0); del buf265  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (2048, 576), (576, 1), 0), permute_214, out=buf267)
    del permute_214
    buf268 = empty((576, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (576, 2048), (1, 576), 0), view_29, out=buf268)
    del view_29
    buf269 = empty((1, 576), device='cpu', dtype=torch.float32)
    buf270 = buf255; del buf255  # reuse
    buf271 = buf254; del buf254  # reuse
    buf272 = empty((192, ), device='cpu', dtype=torch.float32)
    buf273 = empty((192, ), device='cpu', dtype=torch.float32)
    buf274 = reinterpret_tensor(buf264, (8, 192, 16, 16), (49152, 256, 16, 1), 0); del buf264  # reuse
    cpp_fused_convolution_backward_native_layer_norm_backward_sum_45(c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(mul_190.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del buf258
    del buf267
    del buf270
    del buf271
    del div_16
    del mul_190
    del primals_116
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf275 = aten.convolution_backward(buf274, mul_189, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf274
    del mul_189
    del primals_115
    buf276 = buf275[0]
    buf277 = buf275[1]
    del buf275
    buf278 = buf152; del buf152  # reuse
    buf279 = empty((128, ), device='cpu', dtype=torch.float32)
    buf280 = empty((128, ), device='cpu', dtype=torch.float32)
    buf281 = buf276; del buf276  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_46(c_void_p(buf281.data_ptr()), c_void_p(mul_539.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_238.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    del convolution_23
    del mul_539
    del primals_45
    del squeeze_67
    del unsqueeze_238
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf282 = aten.convolution_backward(buf281, add_125, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_125
    del buf281
    del primals_114
    buf283 = buf282[0]
    buf284 = buf282[1]
    del buf282
    buf285 = buf279; del buf279  # reuse
    buf286 = empty((128, ), device='cpu', dtype=torch.float32)
    buf287 = empty((128, ), device='cpu', dtype=torch.float32)
    buf288 = buf283; del buf283  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_47(c_void_p(buf288.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del buf149
    del convolution_22
    del primals_43
    del squeeze_64
    del unsqueeze_250
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf289 = aten.convolution_backward(buf288, mul_174, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf288
    del mul_174
    del primals_113
    buf290 = buf289[0]
    buf291 = buf289[1]
    del buf289
    buf292 = empty((384, ), device='cpu', dtype=torch.float32)
    buf293 = empty((384, ), device='cpu', dtype=torch.float32)
    buf294 = empty((384, ), device='cpu', dtype=torch.float32)
    buf295 = buf290; del buf290  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_48(c_void_p(buf295.data_ptr()), c_void_p(mul_560.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_262.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del convolution_21
    del mul_560
    del primals_41
    del squeeze_61
    del unsqueeze_262
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf296 = aten.convolution_backward(buf295, mul_166, primals_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
    del buf295
    del mul_166
    del primals_112
    buf297 = buf296[0]
    buf298 = buf296[1]
    del buf296
    buf299 = buf293; del buf293  # reuse
    buf300 = empty((384, ), device='cpu', dtype=torch.float32)
    buf301 = empty((384, ), device='cpu', dtype=torch.float32)
    buf302 = buf297; del buf297  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_49(c_void_p(buf302.data_ptr()), c_void_p(mul_572.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del buf300
    del convolution_20
    del mul_572
    del primals_39
    del squeeze_58
    del unsqueeze_274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf303 = aten.convolution_backward(buf302, mul_158, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf302
    del mul_158
    del primals_111
    buf304 = buf303[0]
    buf305 = buf303[1]
    del buf303
    buf306 = empty((96, ), device='cpu', dtype=torch.float32)
    buf307 = empty((96, ), device='cpu', dtype=torch.float32)
    buf308 = empty((96, ), device='cpu', dtype=torch.float32)
    buf309 = buf304; del buf304  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_50(c_void_p(buf309.data_ptr()), c_void_p(mul_584.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_286.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del convolution_19
    del mul_584
    del primals_37
    del squeeze_55
    del unsqueeze_286
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf310 = aten.convolution_backward(buf309, cat, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat
    del primals_110
    buf311 = buf310[0]
    buf312 = buf310[1]
    del buf310
    buf313 = buf307; del buf307  # reuse
    buf314 = empty((96, ), device='cpu', dtype=torch.float32)
    buf315 = empty((96, ), device='cpu', dtype=torch.float32)
    buf316 = buf309; del buf309  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_51(c_void_p(buf311.data_ptr()), c_void_p(mul_596.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    del convolution_18
    del mul_596
    del primals_35
    del squeeze_52
    del unsqueeze_298
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf317 = aten.convolution_backward(buf316, view_25, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf316
    del primals_109
    del view_25
    buf318 = buf317[0]
    buf319 = buf317[1]
    del buf317
    buf320 = empty_strided((32, 256, 1), (1, 32, 8192), device='cpu', dtype=torch.float32)
    buf321 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf322 = reinterpret_tensor(buf266, (32, 256, 144), (36864, 144, 1), 0); del buf266  # reuse
    buf323 = empty((144, ), device='cpu', dtype=torch.float32)
    buf324 = empty((144, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_52(c_void_p(buf318.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(mul_141.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()))
    del div_17
    del mul_141
    del primals_107
    buf325 = empty((8192, 288), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (8192, 144), (144, 1), 0), permute_227, out=buf325)
    del permute_227
    buf326 = empty((144, 288), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (144, 8192), (1, 144), 0), view_21, out=buf326)
    del view_21
    buf327 = empty((1, 144), device='cpu', dtype=torch.float32)
    buf328 = reinterpret_tensor(buf325, (32, 256, 288), (73728, 288, 1), 0); del buf325  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_53(c_void_p(buf328.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf327.data_ptr()))
    del addmm_6
    buf329 = reinterpret_tensor(buf318, (8192, 144), (144, 1), 0); del buf318  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (8192, 288), (288, 1), 0), permute_232, out=buf329)
    del permute_232
    buf330 = empty((288, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (288, 8192), (1, 288), 0), view_19, out=buf330)
    del view_19
    buf331 = empty((1, 288), device='cpu', dtype=torch.float32)
    buf332 = buf321; del buf321  # reuse
    buf333 = reinterpret_tensor(buf320, (32, 256, 1), (256, 1, 8192), 0); del buf320  # reuse
    buf334 = empty((144, ), device='cpu', dtype=torch.float32)
    buf335 = empty((144, ), device='cpu', dtype=torch.float32)
    buf336 = buf322; del buf322  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_54(c_void_p(buf336.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(mul_138.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    del div_18
    del mul_138
    del primals_101
    buf337 = buf329; del buf329  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (8192, 144), (144, 1), 0), permute_236, out=buf337)
    del permute_236
    buf338 = empty((144, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (144, 8192), (1, 144), 0), view_17, out=buf338)
    del view_17
    buf339 = empty((1, 144), device='cpu', dtype=torch.float32)
    cpp_fused_sum_55(c_void_p(buf336.data_ptr()), c_void_p(buf339.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf340 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf337, (32, 4, 256, 36), (36864, 36, 144, 1), 0), getitem_52, getitem_53, getitem_54, alias_16, getitem_56, getitem_57, getitem_58, 0, 0, 0.0, False, getitem_61, getitem_62)
    del alias_16
    del buf337
    del getitem_52
    del getitem_53
    del getitem_54
    del getitem_56
    del getitem_57
    del getitem_58
    del getitem_61
    del getitem_62
    buf341 = buf340[0]
    buf342 = buf340[1]
    buf343 = buf340[2]
    del buf340
    buf344 = empty((32, 256, 3, 4, 36), device='cpu', dtype=torch.float32)
    cpp_fused_clone_56(c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del buf341
    del buf342
    buf345 = reinterpret_tensor(buf343, (8192, 144), (144, 1), 0); del buf343  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (8192, 432), (432, 1), 0), permute_242, out=buf345)
    del permute_242
    buf346 = empty((432, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (432, 8192), (1, 432), 0), view_13, out=buf346)
    del view_13
    buf347 = empty((1, 432), device='cpu', dtype=torch.float32)
    buf348 = buf333; del buf333  # reuse
    buf349 = buf332; del buf332  # reuse
    buf350 = empty((144, ), device='cpu', dtype=torch.float32)
    buf351 = empty((144, ), device='cpu', dtype=torch.float32)
    buf352 = buf336; del buf336  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_57(c_void_p(buf352.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(mul_136.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    del div_19
    del mul_136
    del primals_95
    buf353 = reinterpret_tensor(buf328, (8192, 288), (288, 1), 0); del buf328  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (8192, 144), (144, 1), 0), permute_246, out=buf353)
    del permute_246
    buf354 = empty((144, 288), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (144, 8192), (1, 144), 0), view_11, out=buf354)
    del view_11
    buf355 = empty((1, 144), device='cpu', dtype=torch.float32)
    buf356 = reinterpret_tensor(buf353, (32, 256, 288), (73728, 288, 1), 0); del buf353  # reuse
    cpp_fused_add_fill_mul_silu_sub_sum_58(c_void_p(buf356.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf355.data_ptr()))
    del addmm_2
    buf357 = buf345; del buf345  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf356, (8192, 288), (288, 1), 0), permute_251, out=buf357)
    del permute_251
    buf358 = empty((288, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf356, (288, 8192), (1, 288), 0), view_9, out=buf358)
    del view_9
    buf359 = empty((1, 288), device='cpu', dtype=torch.float32)
    buf360 = buf349; del buf349  # reuse
    buf361 = buf348; del buf348  # reuse
    buf362 = empty((144, ), device='cpu', dtype=torch.float32)
    buf363 = empty((144, ), device='cpu', dtype=torch.float32)
    buf364 = buf352; del buf352  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_59(c_void_p(buf364.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(mul_133.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    del buf356
    del div_20
    del mul_133
    del primals_89
    buf365 = buf357; del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (8192, 144), (144, 1), 0), permute_255, out=buf365)
    del permute_255
    buf366 = empty((144, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (144, 8192), (1, 144), 0), view_7, out=buf366)
    del view_7
    buf367 = empty((1, 144), device='cpu', dtype=torch.float32)
    cpp_fused_sum_60(c_void_p(buf364.data_ptr()), c_void_p(buf367.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf368 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf365, (32, 4, 256, 36), (36864, 36, 144, 1), 0), getitem_36, getitem_37, getitem_38, alias_17, getitem_40, getitem_41, getitem_42, 0, 0, 0.0, False, getitem_45, getitem_46)
    del alias_17
    del buf365
    del getitem_36
    del getitem_37
    del getitem_38
    del getitem_40
    del getitem_41
    del getitem_42
    del getitem_45
    del getitem_46
    buf369 = buf368[0]
    buf370 = buf368[1]
    buf371 = buf368[2]
    del buf368
    buf372 = buf344; del buf344  # reuse
    cpp_fused_clone_61(c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    del buf369
    buf373 = reinterpret_tensor(buf371, (8192, 144), (144, 1), 0); del buf371  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (8192, 432), (432, 1), 0), permute_261, out=buf373)
    del permute_261
    buf374 = empty((432, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (432, 8192), (1, 432), 0), view_3, out=buf374)
    del view_3
    buf375 = empty((1, 432), device='cpu', dtype=torch.float32)
    buf376 = buf361; del buf361  # reuse
    buf377 = buf360; del buf360  # reuse
    buf378 = empty((144, ), device='cpu', dtype=torch.float32)
    buf379 = empty((144, ), device='cpu', dtype=torch.float32)
    buf380 = reinterpret_tensor(buf370, (8, 144, 32, 32), (147456, 1024, 32, 1), 0); del buf370  # reuse
    cpp_fused_convolution_backward_native_layer_norm_backward_sum_62(c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(mul_131.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del buf364
    del buf372
    del buf373
    del buf376
    del buf377
    del div_21
    del mul_131
    del primals_83
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf381 = aten.convolution_backward(buf380, mul_130, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf380
    del mul_130
    del primals_82
    buf382 = buf381[0]
    buf383 = buf381[1]
    del buf381
    buf384 = buf314; del buf314  # reuse
    buf385 = empty((96, ), device='cpu', dtype=torch.float32)
    buf386 = empty((96, ), device='cpu', dtype=torch.float32)
    buf387 = buf382; del buf382  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_63(c_void_p(buf387.data_ptr()), c_void_p(mul_649.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del convolution_16
    del mul_649
    del primals_33
    del squeeze_49
    del unsqueeze_310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf388 = aten.convolution_backward(buf387, add_81, primals_81, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_81
    del buf387
    del primals_81
    buf389 = buf388[0]
    buf390 = buf388[1]
    del buf388
    buf391 = buf385; del buf385  # reuse
    buf392 = empty((96, ), device='cpu', dtype=torch.float32)
    buf393 = empty((96, ), device='cpu', dtype=torch.float32)
    buf394 = buf389; del buf389  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_64(c_void_p(buf394.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()))
    del buf311
    del buf392
    del convolution_15
    del primals_31
    del squeeze_46
    del unsqueeze_322
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf395 = aten.convolution_backward(buf394, mul_115, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf394
    del mul_115
    del primals_80
    buf396 = buf395[0]
    buf397 = buf395[1]
    del buf395
    buf398 = empty((256, ), device='cpu', dtype=torch.float32)
    buf399 = empty((256, ), device='cpu', dtype=torch.float32)
    buf400 = empty((256, ), device='cpu', dtype=torch.float32)
    buf401 = buf396; del buf396  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_65(c_void_p(buf401.data_ptr()), c_void_p(mul_670.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()))
    del convolution_14
    del mul_670
    del primals_29
    del squeeze_43
    del unsqueeze_334
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf402 = aten.convolution_backward(buf401, mul_107, primals_79, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False])
    del mul_107
    del primals_79
    buf403 = buf402[0]
    buf404 = buf402[1]
    del buf402
    buf405 = buf399; del buf399  # reuse
    buf406 = empty((256, ), device='cpu', dtype=torch.float32)
    buf407 = empty((256, ), device='cpu', dtype=torch.float32)
    buf408 = buf403; del buf403  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_66(c_void_p(buf408.data_ptr()), c_void_p(mul_682.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    del convolution_13
    del mul_682
    del primals_27
    del squeeze_40
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf409 = aten.convolution_backward(buf408, add_66, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_66
    del buf408
    del primals_78
    buf410 = buf409[0]
    buf411 = buf409[1]
    del buf409
    buf412 = empty((64, ), device='cpu', dtype=torch.float32)
    buf413 = empty((64, ), device='cpu', dtype=torch.float32)
    buf414 = empty((64, ), device='cpu', dtype=torch.float32)
    buf415 = reinterpret_tensor(buf401, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf401  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_67(c_void_p(buf410.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()))
    del convolution_12
    del primals_25
    del squeeze_37
    del unsqueeze_358
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf416 = aten.convolution_backward(buf415, mul_92, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_92
    del primals_77
    buf417 = buf416[0]
    buf418 = buf416[1]
    del buf416
    buf419 = buf406; del buf406  # reuse
    buf420 = empty((256, ), device='cpu', dtype=torch.float32)
    buf421 = empty((256, ), device='cpu', dtype=torch.float32)
    buf422 = buf417; del buf417  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_68(c_void_p(buf422.data_ptr()), c_void_p(mul_703.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    del convolution_11
    del mul_703
    del primals_23
    del squeeze_34
    del unsqueeze_370
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf423 = aten.convolution_backward(buf422, mul_84, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False])
    del buf422
    del mul_84
    del primals_76
    buf424 = buf423[0]
    buf425 = buf423[1]
    del buf423
    buf426 = buf420; del buf420  # reuse
    buf427 = empty((256, ), device='cpu', dtype=torch.float32)
    buf428 = empty((256, ), device='cpu', dtype=torch.float32)
    buf429 = buf424; del buf424  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_69(c_void_p(buf429.data_ptr()), c_void_p(mul_715.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()))
    del convolution_10
    del mul_715
    del primals_21
    del squeeze_31
    del unsqueeze_382
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf430 = aten.convolution_backward(buf429, add_50, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_50
    del buf429
    del primals_75
    buf431 = buf430[0]
    buf432 = buf430[1]
    del buf430
    buf433 = buf413; del buf413  # reuse
    buf434 = empty((64, ), device='cpu', dtype=torch.float32)
    buf435 = empty((64, ), device='cpu', dtype=torch.float32)
    buf436 = buf415; del buf415  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_70(c_void_p(buf410.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del convolution_9
    del primals_19
    del squeeze_28
    del unsqueeze_394
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf437 = aten.convolution_backward(buf436, mul_69, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf436
    del mul_69
    del primals_74
    buf438 = buf437[0]
    buf439 = buf437[1]
    del buf437
    buf440 = buf427; del buf427  # reuse
    buf441 = empty((256, ), device='cpu', dtype=torch.float32)
    buf442 = empty((256, ), device='cpu', dtype=torch.float32)
    buf443 = buf438; del buf438  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_71(c_void_p(buf443.data_ptr()), c_void_p(mul_736.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    del convolution_8
    del mul_736
    del primals_17
    del squeeze_25
    del unsqueeze_406
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf444 = aten.convolution_backward(buf443, mul_61, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False])
    del buf443
    del mul_61
    del primals_73
    buf445 = buf444[0]
    buf446 = buf444[1]
    del buf444
    buf447 = buf441; del buf441  # reuse
    buf448 = empty((256, ), device='cpu', dtype=torch.float32)
    buf449 = empty((256, ), device='cpu', dtype=torch.float32)
    buf450 = buf445; del buf445  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_72(c_void_p(buf450.data_ptr()), c_void_p(mul_748.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()))
    del buf448
    del convolution_7
    del mul_748
    del primals_15
    del squeeze_22
    del unsqueeze_418
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf451 = aten.convolution_backward(buf450, add_34, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_34
    del buf450
    del primals_72
    buf452 = buf451[0]
    buf453 = buf451[1]
    del buf451
    buf454 = buf434; del buf434  # reuse
    buf455 = empty((64, ), device='cpu', dtype=torch.float32)
    buf456 = buf410; del buf410  # reuse
    buf457 = buf455; del buf455  # reuse
    cpp_fused_add_native_batch_norm_backward_73(c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf454.data_ptr()))
    del buf431
    del buf452
    del convolution_6
    del primals_13
    del squeeze_19
    del unsqueeze_430
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf458 = aten.convolution_backward(buf456, mul_46, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf456
    del mul_46
    del primals_71
    buf459 = buf458[0]
    buf460 = buf458[1]
    del buf458
    buf461 = buf286; del buf286  # reuse
    buf462 = empty((128, ), device='cpu', dtype=torch.float32)
    buf463 = empty((128, ), device='cpu', dtype=torch.float32)
    buf464 = buf459; del buf459  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_74(c_void_p(buf464.data_ptr()), c_void_p(mul_769.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    del convolution_5
    del mul_769
    del primals_11
    del squeeze_16
    del unsqueeze_442
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf465 = aten.convolution_backward(buf464, mul_38, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf464
    del mul_38
    del primals_70
    buf466 = buf465[0]
    buf467 = buf465[1]
    del buf465
    buf468 = buf462; del buf462  # reuse
    buf469 = empty((128, ), device='cpu', dtype=torch.float32)
    buf470 = empty((128, ), device='cpu', dtype=torch.float32)
    buf471 = buf466; del buf466  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_75(c_void_p(buf471.data_ptr()), c_void_p(mul_781.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()))
    del buf469
    del convolution_4
    del mul_781
    del primals_9
    del squeeze_13
    del unsqueeze_454
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf472 = aten.convolution_backward(buf471, add_19, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_19
    del buf471
    del primals_69
    buf473 = buf472[0]
    buf474 = buf472[1]
    del buf472
    buf475 = empty((32, ), device='cpu', dtype=torch.float32)
    buf476 = empty((32, ), device='cpu', dtype=torch.float32)
    buf477 = empty((32, ), device='cpu', dtype=torch.float32)
    buf478 = buf473; del buf473  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_76(c_void_p(buf478.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del buf476
    del convolution_3
    del primals_7
    del squeeze_10
    del unsqueeze_466
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf479 = aten.convolution_backward(buf478, mul_23, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf478
    del mul_23
    del primals_68
    buf480 = buf479[0]
    buf481 = buf479[1]
    del buf479
    buf482 = empty((64, ), device='cpu', dtype=torch.float32)
    buf483 = empty((64, ), device='cpu', dtype=torch.float32)
    buf484 = empty((64, ), device='cpu', dtype=torch.float32)
    buf485 = buf480; del buf480  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_77(c_void_p(buf485.data_ptr()), c_void_p(mul_802.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()))
    del convolution_2
    del mul_802
    del primals_5
    del squeeze_7
    del unsqueeze_478
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf486 = aten.convolution_backward(buf485, mul_15, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
    del buf485
    del mul_15
    del primals_67
    buf487 = buf486[0]
    buf488 = buf486[1]
    del buf486
    buf489 = buf483; del buf483  # reuse
    buf490 = empty((64, ), device='cpu', dtype=torch.float32)
    buf491 = empty((64, ), device='cpu', dtype=torch.float32)
    buf492 = buf487; del buf487  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_78(c_void_p(buf492.data_ptr()), c_void_p(mul_814.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()))
    del buf490
    del convolution_1
    del mul_814
    del primals_3
    del squeeze_4
    del unsqueeze_490
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf493 = aten.convolution_backward(buf492, mul_7, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf492
    del mul_7
    del primals_66
    buf494 = buf493[0]
    buf495 = buf493[1]
    del buf493
    buf496 = empty((16, ), device='cpu', dtype=torch.float32)
    buf497 = empty((16, ), device='cpu', dtype=torch.float32)
    buf498 = empty((16, ), device='cpu', dtype=torch.float32)
    buf499 = buf494; del buf494  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_79(c_void_p(buf499.data_ptr()), c_void_p(mul_826.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()))
    del buf497
    del convolution
    del mul_826
    del primals_1
    del squeeze_1
    del unsqueeze_502
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf500 = aten.convolution_backward(buf499, primals_312, primals_65, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf499
    del primals_312
    del primals_65
    buf501 = buf500[1]
    return (buf498, buf496, buf491, buf489, buf484, buf482, buf477, buf475, buf470, buf468, buf463, buf461, buf457, buf454, buf449, buf447, buf442, buf440, buf435, buf433, buf428, buf426, buf421, buf419, buf414, buf412, buf407, buf405, buf400, buf398, buf393, buf391, buf386, buf384, buf315, buf313, buf308, buf306, buf301, buf299, buf294, buf292, buf287, buf285, buf280, buf278, buf153, buf151, buf146, buf144, buf139, buf137, buf132, buf130, buf125, buf123, buf118, buf116, buf19, buf17, buf12, buf10, buf5, buf3, buf501, buf495, buf488, buf481, buf474, buf467, buf460, buf453, buf446, buf439, buf432, buf425, buf418, buf411, buf404, buf397, buf390, buf383, buf378, buf379, reinterpret_tensor(buf374, (432, 144), (144, 1), 0), reinterpret_tensor(buf375, (432, ), (1, ), 0), reinterpret_tensor(buf366, (144, 144), (144, 1), 0), reinterpret_tensor(buf367, (144, ), (1, ), 0), buf362, buf363, reinterpret_tensor(buf358, (288, 144), (144, 1), 0), reinterpret_tensor(buf359, (288, ), (1, ), 0), reinterpret_tensor(buf354, (144, 288), (288, 1), 0), reinterpret_tensor(buf355, (144, ), (1, ), 0), buf350, buf351, reinterpret_tensor(buf346, (432, 144), (144, 1), 0), reinterpret_tensor(buf347, (432, ), (1, ), 0), reinterpret_tensor(buf338, (144, 144), (144, 1), 0), reinterpret_tensor(buf339, (144, ), (1, ), 0), buf334, buf335, reinterpret_tensor(buf330, (288, 144), (144, 1), 0), reinterpret_tensor(buf331, (288, ), (1, ), 0), reinterpret_tensor(buf326, (144, 288), (288, 1), 0), reinterpret_tensor(buf327, (144, ), (1, ), 0), buf323, buf324, buf319, buf312, buf305, buf298, buf291, buf284, buf277, buf272, buf273, reinterpret_tensor(buf268, (576, 192), (192, 1), 0), reinterpret_tensor(buf269, (576, ), (1, ), 0), reinterpret_tensor(buf260, (192, 192), (192, 1), 0), reinterpret_tensor(buf261, (192, ), (1, ), 0), buf256, buf257, reinterpret_tensor(buf252, (384, 192), (192, 1), 0), reinterpret_tensor(buf253, (384, ), (1, ), 0), reinterpret_tensor(buf248, (192, 384), (384, 1), 0), reinterpret_tensor(buf249, (192, ), (1, ), 0), buf244, buf245, reinterpret_tensor(buf240, (576, 192), (192, 1), 0), reinterpret_tensor(buf241, (576, ), (1, ), 0), reinterpret_tensor(buf232, (192, 192), (192, 1), 0), reinterpret_tensor(buf233, (192, ), (1, ), 0), buf228, buf229, reinterpret_tensor(buf224, (384, 192), (192, 1), 0), reinterpret_tensor(buf225, (384, ), (1, ), 0), reinterpret_tensor(buf220, (192, 384), (384, 1), 0), reinterpret_tensor(buf221, (192, ), (1, ), 0), buf216, buf217, reinterpret_tensor(buf212, (576, 192), (192, 1), 0), reinterpret_tensor(buf213, (576, ), (1, ), 0), reinterpret_tensor(buf204, (192, 192), (192, 1), 0), reinterpret_tensor(buf205, (192, ), (1, ), 0), buf200, buf201, reinterpret_tensor(buf196, (384, 192), (192, 1), 0), reinterpret_tensor(buf197, (384, ), (1, ), 0), reinterpret_tensor(buf192, (192, 384), (384, 1), 0), reinterpret_tensor(buf193, (192, ), (1, ), 0), buf188, buf189, reinterpret_tensor(buf184, (576, 192), (192, 1), 0), reinterpret_tensor(buf185, (576, ), (1, ), 0), reinterpret_tensor(buf176, (192, 192), (192, 1), 0), reinterpret_tensor(buf177, (192, ), (1, ), 0), buf172, buf173, reinterpret_tensor(buf168, (384, 192), (192, 1), 0), reinterpret_tensor(buf169, (384, ), (1, ), 0), reinterpret_tensor(buf164, (192, 384), (384, 1), 0), reinterpret_tensor(buf165, (192, ), (1, ), 0), buf161, buf162, buf157, buf150, buf143, buf136, buf129, buf122, buf115, buf110, buf111, reinterpret_tensor(buf106, (720, 240), (240, 1), 0), reinterpret_tensor(buf107, (720, ), (1, ), 0), reinterpret_tensor(buf98, (240, 240), (240, 1), 0), reinterpret_tensor(buf99, (240, ), (1, ), 0), buf94, buf95, reinterpret_tensor(buf90, (480, 240), (240, 1), 0), reinterpret_tensor(buf91, (480, ), (1, ), 0), reinterpret_tensor(buf86, (240, 480), (480, 1), 0), reinterpret_tensor(buf87, (240, ), (1, ), 0), buf82, buf83, reinterpret_tensor(buf78, (720, 240), (240, 1), 0), reinterpret_tensor(buf79, (720, ), (1, ), 0), reinterpret_tensor(buf70, (240, 240), (240, 1), 0), reinterpret_tensor(buf71, (240, ), (1, ), 0), buf66, buf67, reinterpret_tensor(buf62, (480, 240), (240, 1), 0), reinterpret_tensor(buf63, (480, ), (1, ), 0), reinterpret_tensor(buf58, (240, 480), (480, 1), 0), reinterpret_tensor(buf59, (240, ), (1, ), 0), buf54, buf55, reinterpret_tensor(buf50, (720, 240), (240, 1), 0), reinterpret_tensor(buf51, (720, ), (1, ), 0), reinterpret_tensor(buf42, (240, 240), (240, 1), 0), reinterpret_tensor(buf43, (240, ), (1, ), 0), buf38, buf39, reinterpret_tensor(buf34, (480, 240), (240, 1), 0), reinterpret_tensor(buf35, (480, ), (1, ), 0), reinterpret_tensor(buf30, (240, 480), (480, 1), 0), reinterpret_tensor(buf31, (240, ), (1, ), 0), buf27, buf28, buf23, buf16, buf9, reinterpret_tensor(buf1, (1000, 640), (640, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((144, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((96, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((96, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((160, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((160, 320, 3, 3), (2880, 1, 960, 320), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 16, 128, 128), (262144, 1, 2048, 16), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 16, 128, 128), (262144, 1, 2048, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_15 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_19 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_38 = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_46 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_34 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_61 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_69 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_50 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_84 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_92 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_66 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_107 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_115 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    add_81 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    mul_130 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    mul_131 = rand_strided((32, 256, 144), (36864, 144, 1), device='cpu', dtype=torch.float32)
    view_3 = rand_strided((8192, 144), (144, 1), device='cpu', dtype=torch.float32)
    getitem_36 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cpu', dtype=torch.float32)
    getitem_38 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cpu', dtype=torch.float32)
    getitem_40 = rand_strided((32, 4, 256), (1024, 1, 4), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_42 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_45 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_46 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_7 = rand_strided((8192, 144), (144, 1), device='cpu', dtype=torch.float32)
    mul_133 = rand_strided((32, 256, 144), (36864, 144, 1), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((8192, 144), (144, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((8192, 288), (288, 1), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((8192, 288), (288, 1), device='cpu', dtype=torch.float32)
    mul_136 = rand_strided((32, 256, 144), (36864, 144, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((8192, 144), (144, 1), device='cpu', dtype=torch.float32)
    getitem_52 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cpu', dtype=torch.float32)
    getitem_54 = rand_strided((32, 4, 256, 36), (110592, 36, 432, 1), device='cpu', dtype=torch.float32)
    getitem_56 = rand_strided((32, 4, 256), (1024, 1, 4), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_58 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_61 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_62 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_17 = rand_strided((8192, 144), (144, 1), device='cpu', dtype=torch.float32)
    mul_138 = rand_strided((32, 256, 144), (36864, 144, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((8192, 144), (144, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((8192, 288), (288, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((8192, 288), (288, 1), device='cpu', dtype=torch.float32)
    mul_141 = rand_strided((32, 256, 144), (36864, 144, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((8, 144, 32, 32), (147456, 1, 4608, 144), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    mul_158 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 384, 32, 32), (393216, 1, 12288, 384), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    mul_166 = rand_strided((8, 384, 32, 32), (393216, 1, 12288, 384), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 16, 16), (98304, 1, 6144, 384), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    mul_174 = rand_strided((8, 384, 16, 16), (98304, 1, 6144, 384), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_125 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_189 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    mul_190 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    getitem_82 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_84 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_86 = rand_strided((32, 4, 64), (256, 1, 4), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_88 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_91 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_92 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_33 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_195 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    getitem_98 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_100 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_102 = rand_strided((32, 4, 64), (256, 1, 4), device='cpu', dtype=torch.float32)
    getitem_103 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_104 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_107 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_108 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_43 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_197 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_200 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    getitem_114 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_115 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_116 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_118 = rand_strided((32, 4, 64), (256, 1, 4), device='cpu', dtype=torch.float32)
    getitem_119 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_120 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_123 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_124 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_53 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_202 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_205 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    getitem_130 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_132 = rand_strided((32, 4, 64, 48), (36864, 48, 576, 1), device='cpu', dtype=torch.float32)
    getitem_134 = rand_strided((32, 4, 64), (256, 1, 4), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_136 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_139 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_140 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_63 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    mul_207 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((2048, 192), (192, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((2048, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_210 = rand_strided((32, 64, 192), (12288, 192, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((8, 192, 16, 16), (49152, 1, 3072, 192), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_227 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_235 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_243 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_181 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    mul_258 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    mul_259 = rand_strided((32, 16, 240), (3840, 240, 1), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    getitem_160 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_161 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_162 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_164 = rand_strided((32, 4, 16), (64, 1, 4), device='cpu', dtype=torch.float32)
    getitem_165 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_166 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_169 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_170 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_79 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    mul_261 = rand_strided((32, 16, 240), (3840, 240, 1), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((512, 480), (480, 1), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((512, 480), (480, 1), device='cpu', dtype=torch.float32)
    mul_264 = rand_strided((32, 16, 240), (3840, 240, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    getitem_176 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_177 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_178 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_180 = rand_strided((32, 4, 16), (64, 1, 4), device='cpu', dtype=torch.float32)
    getitem_181 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_182 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_185 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_186 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_89 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    mul_266 = rand_strided((32, 16, 240), (3840, 240, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((512, 480), (480, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((512, 480), (480, 1), device='cpu', dtype=torch.float32)
    mul_269 = rand_strided((32, 16, 240), (3840, 240, 1), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    getitem_192 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_193 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_194 = rand_strided((32, 4, 16, 60), (11520, 60, 720, 1), device='cpu', dtype=torch.float32)
    getitem_196 = rand_strided((32, 4, 16), (64, 1, 4), device='cpu', dtype=torch.float32)
    getitem_197 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_198 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_201 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_202 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_99 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    mul_271 = rand_strided((32, 16, 240), (3840, 240, 1), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((512, 240), (240, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((512, 480), (480, 1), device='cpu', dtype=torch.float32)
    view_103 = rand_strided((512, 480), (480, 1), device='cpu', dtype=torch.float32)
    mul_274 = rand_strided((32, 16, 240), (3840, 240, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((8, 240, 8, 8), (15360, 1, 1920, 240), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    mul_291 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    clone_64 = rand_strided((8, 640), (640, 1), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((1000, 640), (640, 1), device='cpu', dtype=torch.float32)
    mul_301 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_313 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    unsqueeze_142 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_325 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((32, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    permute_76 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    permute_81 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((32, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    permute_85 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    alias_9 = rand_strided((32, 4, 16, 60), (3840, 1, 240, 4), device='cpu', dtype=torch.float32)
    permute_91 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((32, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    permute_95 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((32, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    alias_10 = rand_strided((32, 4, 16, 60), (3840, 1, 240, 4), device='cpu', dtype=torch.float32)
    permute_110 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((32, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    permute_114 = rand_strided((240, 480), (480, 1), device='cpu', dtype=torch.float32)
    permute_119 = rand_strided((480, 240), (240, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((32, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((240, 240), (240, 1), device='cpu', dtype=torch.float32)
    alias_11 = rand_strided((32, 4, 16, 60), (3840, 1, 240, 4), device='cpu', dtype=torch.float32)
    permute_129 = rand_strided((720, 240), (240, 1), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((32, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    mul_395 = rand_strided((8, 160, 8, 8), (10240, 1, 1280, 160), device='cpu', dtype=torch.float32)
    unsqueeze_166 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_416 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_428 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_440 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_452 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_147 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((32, 4, 64, 48), (12288, 1, 192, 4), device='cpu', dtype=torch.float32)
    permute_157 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((32, 4, 64, 48), (12288, 1, 192, 4), device='cpu', dtype=torch.float32)
    permute_176 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_180 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_185 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((32, 4, 64, 48), (12288, 1, 192, 4), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((192, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((384, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((32, 4, 64, 48), (12288, 1, 192, 4), device='cpu', dtype=torch.float32)
    permute_214 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((32, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    mul_539 = rand_strided((8, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_560 = rand_strided((8, 384, 16, 16), (98304, 1, 6144, 384), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_572 = rand_strided((8, 384, 32, 32), (393216, 1, 12288, 384), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_584 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_596 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((32, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((144, 288), (288, 1), device='cpu', dtype=torch.float32)
    permute_232 = rand_strided((288, 144), (144, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((32, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((144, 144), (144, 1), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((32, 4, 256, 36), (36864, 1, 144, 4), device='cpu', dtype=torch.float32)
    permute_242 = rand_strided((432, 144), (144, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((32, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    permute_246 = rand_strided((144, 288), (288, 1), device='cpu', dtype=torch.float32)
    permute_251 = rand_strided((288, 144), (144, 1), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((32, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((144, 144), (144, 1), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((32, 4, 256, 36), (36864, 1, 144, 4), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((432, 144), (144, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((32, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    mul_649 = rand_strided((8, 96, 32, 32), (98304, 1, 3072, 96), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_670 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_682 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_703 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_715 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_736 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_748 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_769 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_781 = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_802 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_814 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_826 = rand_strided((8, 16, 128, 128), (262144, 1, 2048, 16), device='cpu', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_89, primals_95, primals_101, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_152, primals_158, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_211, primals_212, primals_213, primals_312, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, mul_38, convolution_5, squeeze_16, mul_46, convolution_6, squeeze_19, add_34, convolution_7, squeeze_22, mul_61, convolution_8, squeeze_25, mul_69, convolution_9, squeeze_28, add_50, convolution_10, squeeze_31, mul_84, convolution_11, squeeze_34, mul_92, convolution_12, squeeze_37, add_66, convolution_13, squeeze_40, mul_107, convolution_14, squeeze_43, mul_115, convolution_15, squeeze_46, add_81, convolution_16, squeeze_49, mul_130, mul_131, view_3, getitem_36, getitem_37, getitem_38, getitem_40, getitem_41, getitem_42, getitem_45, getitem_46, view_7, mul_133, view_9, addmm_2, view_11, mul_136, view_13, getitem_52, getitem_53, getitem_54, getitem_56, getitem_57, getitem_58, getitem_61, getitem_62, view_17, mul_138, view_19, addmm_6, view_21, mul_141, view_25, convolution_18, squeeze_52, cat, convolution_19, squeeze_55, mul_158, convolution_20, squeeze_58, mul_166, convolution_21, squeeze_61, mul_174, convolution_22, squeeze_64, add_125, convolution_23, squeeze_67, mul_189, mul_190, view_29, getitem_82, getitem_83, getitem_84, getitem_86, getitem_87, getitem_88, getitem_91, getitem_92, view_33, mul_192, view_35, addmm_10, view_37, mul_195, view_39, getitem_98, getitem_99, getitem_100, getitem_102, getitem_103, getitem_104, getitem_107, getitem_108, view_43, mul_197, view_45, addmm_14, view_47, mul_200, view_49, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, getitem_123, getitem_124, view_53, mul_202, view_55, addmm_18, view_57, mul_205, view_59, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, getitem_139, getitem_140, view_63, mul_207, view_65, addmm_22, view_67, mul_210, view_71, convolution_25, squeeze_70, cat_1, convolution_26, squeeze_73, mul_227, convolution_27, squeeze_76, mul_235, convolution_28, squeeze_79, mul_243, convolution_29, squeeze_82, add_181, convolution_30, squeeze_85, mul_258, mul_259, view_75, getitem_160, getitem_161, getitem_162, getitem_164, getitem_165, getitem_166, getitem_169, getitem_170, view_79, mul_261, view_81, addmm_26, view_83, mul_264, view_85, getitem_176, getitem_177, getitem_178, getitem_180, getitem_181, getitem_182, getitem_185, getitem_186, view_89, mul_266, view_91, addmm_30, view_93, mul_269, view_95, getitem_192, getitem_193, getitem_194, getitem_196, getitem_197, getitem_198, getitem_201, getitem_202, view_99, mul_271, view_101, addmm_34, view_103, mul_274, view_107, convolution_32, squeeze_88, cat_2, convolution_33, squeeze_91, mul_291, convolution_34, squeeze_94, clone_64, permute_67, mul_301, unsqueeze_130, mul_313, unsqueeze_142, mul_325, unsqueeze_154, div_1, permute_76, permute_81, div_2, permute_85, alias_9, permute_91, div_3, permute_95, permute_100, div_4, permute_104, alias_10, permute_110, div_5, permute_114, permute_119, div_6, permute_123, alias_11, permute_129, div_7, mul_395, unsqueeze_166, unsqueeze_178, mul_416, unsqueeze_190, mul_428, unsqueeze_202, mul_440, unsqueeze_214, mul_452, unsqueeze_226, div_8, permute_142, permute_147, div_9, permute_151, alias_12, permute_157, div_10, permute_161, permute_166, div_11, permute_170, alias_13, permute_176, div_12, permute_180, permute_185, div_13, permute_189, alias_14, permute_195, div_14, permute_199, permute_204, div_15, permute_208, alias_15, permute_214, div_16, mul_539, unsqueeze_238, unsqueeze_250, mul_560, unsqueeze_262, mul_572, unsqueeze_274, mul_584, unsqueeze_286, mul_596, unsqueeze_298, div_17, permute_227, permute_232, div_18, permute_236, alias_16, permute_242, div_19, permute_246, permute_251, div_20, permute_255, alias_17, permute_261, div_21, mul_649, unsqueeze_310, unsqueeze_322, mul_670, unsqueeze_334, mul_682, unsqueeze_346, unsqueeze_358, mul_703, unsqueeze_370, mul_715, unsqueeze_382, unsqueeze_394, mul_736, unsqueeze_406, mul_748, unsqueeze_418, unsqueeze_430, mul_769, unsqueeze_442, mul_781, unsqueeze_454, unsqueeze_466, mul_802, unsqueeze_478, mul_814, unsqueeze_490, mul_826, unsqueeze_502, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilevit_s', benchmark_compiled_module)
