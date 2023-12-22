
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(49.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
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
                        tmp15.store(out_ptr3 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(1L + (2L*x0) + (2L*x0_inner) + (464L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(1L + (2L*x1) + (2L*x1_inner) + (464L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = tmp8.rsqrt();
                auto tmp11 = tmp9 * tmp10;
                auto tmp12 = tmp4 * tmp11;
                tmp12.store(out_ptr2 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (232L*x2) + (11368L*x1))];
                        auto tmp26 = in_ptr3[static_cast<long>(x0 + (232L*x2) + (11368L*x1))];
                        auto tmp27 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(1L + (2L*x0));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(232);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(232L))) + (464L*x2) + (22736L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 232L)))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(464);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-11319L) + x2 + (98L*x0) + (11368L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp0 ? tmp16 : tmp15;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(232L))) + (464L*x2) + (22736L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 232L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = [&]
                        {
                            auto tmp22 = in_ptr2[static_cast<long>((-11319L) + x2 + (98L*x0) + (11368L*x1))];
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp24 = tmp5 ? tmp20 : tmp23;
                        auto tmp25 = tmp0 ? tmp16 : tmp24;
                        auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                        auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp29;
                    }
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (232L*x1) + (11368L*x0))];
                    auto tmp18 = in_ptr5[static_cast<long>(x2)];
                    auto tmp22 = in_ptr6[static_cast<long>(x2)];
                    auto tmp1 = c10::convert<long>(1L + (2L*x2));
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(232);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(232L))) + (464L*x1) + (22736L*x0) + (c10::div_floor_integer((1L + (2L*x2)), 232L)))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(464);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-11319L) + x1 + (98L*x2) + (11368L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp0 ? tmp16 : tmp15;
                    auto tmp19 = static_cast<float>(1e-05);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = 1 / std::sqrt(tmp20);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp17)(tmp17 * tmp23);
                    out_ptr2[static_cast<long>(x2 + (232L*x1) + (11368L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (232L*x2) + (11368L*x0))];
                    auto tmp32 = in_ptr4[static_cast<long>(x1)];
                    auto tmp36 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(1L + (2L*x1));
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(232);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = c10::convert<long>((2L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(232L))) + (c10::div_floor_integer((1L + (2L*x1)), 232L)));
                        auto tmp8 = static_cast<long>(0);
                        auto tmp9 = tmp7 >= tmp8;
                        auto tmp10 = static_cast<long>(232);
                        auto tmp11 = tmp7 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(232L))) + (c10::div_floor_integer((1L + (2L*x1)), 232L)))) % static_cast<long>(232L))) + (464L*x2) + (22736L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(232L))) + (c10::div_floor_integer((1L + (2L*x1)), 232L))), 232L)) % static_cast<long>(2L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp7 >= tmp10;
                        auto tmp16 = static_cast<long>(464);
                        auto tmp17 = tmp7 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-11368L) + x2 + (49L*(c10::div_floor_integer((1L + (2L*x1)), 232L))) + (98L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(232L))) + (11368L*x0))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp23 = tmp1 >= tmp4;
                    auto tmp24 = static_cast<long>(464);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr3[static_cast<long>((-11319L) + x2 + (98L*x1) + (11368L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp5 ? tmp22 : tmp28;
                    auto tmp30 = static_cast<float>(0.0);
                    auto tmp31 = tmp0 ? tmp30 : tmp29;
                    auto tmp33 = static_cast<float>(1e-05);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    auto tmp35 = 1 / std::sqrt(tmp34);
                    auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                    auto tmp38 = decltype(tmp31)(tmp31 * tmp37);
                    out_ptr0[static_cast<long>(x2 + (49L*x1) + (11368L*x0))] = tmp31;
                    out_ptr1[static_cast<long>(x1 + (232L*x2) + (11368L*x0))] = tmp38;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (11368L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (11368L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (232L*x2) + (232L*x2_inner) + (11368L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (11368L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (232L*x2) + (11368L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
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
''')


cpp_fused_convolution_backward_native_batch_norm_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(464L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(232);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = c10::convert<int>((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 >= tmp7;
                        auto tmp9 = static_cast<int>(232);
                        auto tmp10 = tmp6 < tmp9;
                        auto tmp12 = tmp10 & tmp4;
                        auto tmp11 = [&]
                        {
                            auto tmp13 = c10::convert<int>((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L)));
                            auto tmp14 = static_cast<int>(0);
                            auto tmp15 = tmp13 >= tmp14;
                            auto tmp16 = static_cast<int>(232);
                            auto tmp17 = tmp13 < tmp16;
                            auto tmp19 = tmp17 & tmp12;
                            auto tmp18 = [&]
                            {
                                auto tmp20 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L)))) % static_cast<long>(232L))) + (464L*x2) + (464L*x2_inner) + (22736L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L))), 232L)) % static_cast<long>(2L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp19));
                            auto tmp22 = tmp13 >= tmp16;
                            auto tmp23 = static_cast<int>(464);
                            auto tmp24 = tmp13 < tmp23;
                            auto tmp26 = tmp22 & tmp12;
                            auto tmp25 = [&]
                            {
                                auto tmp27 = masked_load(in_ptr1 + static_cast<long>((-11368L) + x2 + (49L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L))) + (98L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (11368L*x0)), to_float_mask(tmp26));
                                return tmp27;
                            }
                            ;
                            auto tmp28 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp26));
                            auto tmp29 = to_float_mask(tmp17);
                            auto tmp30 = decltype(tmp21)::blendv(tmp28, tmp21, tmp29);
                            return tmp30;
                        }
                        ;
                        auto tmp31 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp12));
                        auto tmp32 = tmp6 >= tmp9;
                        auto tmp33 = static_cast<int>(464);
                        auto tmp34 = tmp6 < tmp33;
                        auto tmp36 = tmp32 & tmp4;
                        auto tmp35 = [&]
                        {
                            auto tmp37 = masked_load(in_ptr2 + static_cast<long>((-11368L) + x2 + (49L*(c10::div_floor_integer(x1, 232L))) + (98L*(static_cast<long>(x1) % static_cast<long>(232L))) + (11368L*x0)), to_float_mask(tmp36));
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp35())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp35(), to_float_mask(tmp36));
                        auto tmp39 = to_float_mask(tmp10);
                        auto tmp40 = decltype(tmp31)::blendv(tmp38, tmp31, tmp39);
                        return tmp40;
                    }
                    ;
                    auto tmp41 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                    auto tmp42 = tmp0 >= tmp3;
                    auto tmp43 = static_cast<int>(464);
                    auto tmp44 = tmp0 < tmp43;
                    auto tmp45 = [&]
                    {
                        auto tmp46 = masked_load(in_ptr3 + static_cast<long>((-11368L) + x2 + (49L*x1) + (11368L*x0)), to_float_mask(tmp42));
                        return tmp46;
                    }
                    ;
                    auto tmp47 = decltype(tmp45())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp45(), to_float_mask(tmp42));
                    auto tmp48 = to_float_mask(tmp4);
                    auto tmp49 = decltype(tmp41)::blendv(tmp47, tmp41, tmp48);
                    tmp49.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (22736L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(232);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = c10::convert<long>((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)));
                        auto tmp7 = static_cast<long>(0);
                        auto tmp8 = tmp6 >= tmp7;
                        auto tmp9 = static_cast<long>(232);
                        auto tmp10 = tmp6 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = c10::convert<long>((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L)));
                            auto tmp13 = static_cast<long>(0);
                            auto tmp14 = tmp12 >= tmp13;
                            auto tmp15 = static_cast<long>(232);
                            auto tmp16 = tmp12 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L)))) % static_cast<long>(232L))) + (464L*x2) + (22736L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L))), 232L)) % static_cast<long>(2L)))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = tmp12 >= tmp15;
                            auto tmp21 = static_cast<long>(464);
                            auto tmp22 = tmp12 < tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = in_ptr1[static_cast<long>((-11368L) + x2 + (49L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L))), 232L)) % static_cast<long>(2L))) + (98L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(232L))) + (c10::div_floor_integer(x1, 232L)))) % static_cast<long>(232L))) + (11368L*x0))];
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp20 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp26 = tmp16 ? tmp19 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp28 = tmp6 >= tmp9;
                        auto tmp29 = static_cast<long>(464);
                        auto tmp30 = tmp6 < tmp29;
                        auto tmp31 = [&]
                        {
                            auto tmp32 = in_ptr2[static_cast<long>((-11368L) + x2 + (49L*(c10::div_floor_integer(x1, 232L))) + (98L*(static_cast<long>(x1) % static_cast<long>(232L))) + (11368L*x0))];
                            return tmp32;
                        }
                        ;
                        auto tmp33 = tmp28 ? tmp31() : static_cast<decltype(tmp31())>(0.0);
                        auto tmp34 = tmp10 ? tmp27 : tmp33;
                        return tmp34;
                    }
                    ;
                    auto tmp35 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp36 = tmp0 >= tmp3;
                    auto tmp37 = static_cast<long>(464);
                    auto tmp38 = tmp0 < tmp37;
                    auto tmp39 = [&]
                    {
                        auto tmp40 = in_ptr3[static_cast<long>((-11368L) + x2 + (49L*x1) + (11368L*x0))];
                        return tmp40;
                    }
                    ;
                    auto tmp41 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                    auto tmp42 = tmp4 ? tmp35 : tmp41;
                    out_ptr0[static_cast<long>(x2 + (49L*x1) + (22736L*x0))] = tmp42;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(49L + x2 + (98L*x0) + (22736L*x1)), static_cast<long>(98L), tmp1, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(49L + x2 + (98L*x0) + (22736L*x1)), static_cast<long>(98L), tmp1, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x0 + (232L*x2) + (232L*x2_inner) + (11368L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (232L*x2) + (232L*x2_inner) + (11368L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp9 = tmp5 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x0 + (232L*x2) + (11368L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(49L + x2 + (98L*x0) + (98L*x0_inner) + (22736L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (232L*x2) + (11368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(49L + x1 + (98L*x2) + (22736L*x0)), static_cast<long>(98L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x2 + (232L*x1) + (232L*x1_inner) + (11368L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp12 = tmp10 * tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        tmp13.store(out_ptr3 + static_cast<long>(x2 + (232L*x1) + (232L*x1_inner) + (11368L*x0)));
                    }
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x2 + (232L*x1) + (11368L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(49L + x1 + (98L*x2) + (22736L*x0))];
                    auto tmp4 = in_ptr7[static_cast<long>(x2)];
                    auto tmp8 = in_ptr8[static_cast<long>(x2)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                    out_ptr3[static_cast<long>(x2 + (232L*x1) + (11368L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (232L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (232L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (98L*x0) + (22736L*x1)), static_cast<long>(98L), tmp1, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (98L*x0) + (22736L*x1)), static_cast<long>(98L), tmp1, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (232L*x2) + (232L*x2_inner) + (11368L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (232L*x2) + (232L*x2_inner) + (11368L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp9 = tmp5 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (232L*x2) + (11368L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (98L*x0) + (98L*x0_inner) + (22736L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (232L*x2) + (11368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (98L*x2) + (22736L*x0)), static_cast<long>(98L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (232L*x1) + (232L*x1_inner) + (11368L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp12 = tmp10 * tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        tmp13.store(out_ptr2 + static_cast<long>(x2 + (232L*x1) + (232L*x1_inner) + (11368L*x0)));
                    }
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(232L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (232L*x1) + (11368L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (98L*x2) + (22736L*x0))];
                    auto tmp4 = in_ptr4[static_cast<long>(x2)];
                    auto tmp8 = in_ptr5[static_cast<long>(x2)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = 1 / std::sqrt(tmp6);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                    out_ptr2[static_cast<long>(x2 + (232L*x1) + (11368L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (232L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (232L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(232L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (232L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(1L + (2L*x0) + (2L*x0_inner) + (232L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>(1L + (2L*x0) + (2L*x0_inner) + (232L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(1L + (2L*x0) + (232L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(1L + (2L*x0) + (232L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0 + (116L*x1))];
                    auto tmp7 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp0 ? tmp4 : tmp3;
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                    tmp_acc0 = tmp_acc0 + tmp5;
                    tmp_acc1 = tmp_acc1 + tmp9;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(1L + (2L*x1) + (2L*x1_inner) + (232L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(1L + (2L*x1) + (2L*x1_inner) + (232L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp13 = tmp11 * tmp12;
                auto tmp14 = tmp6 * tmp13;
                tmp14.store(out_ptr2 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(1L + (2L*x1) + (232L*x0))];
                auto tmp2 = in_ptr2[static_cast<long>(1L + (2L*x1) + (232L*x0))];
                auto tmp6 = in_ptr5[static_cast<long>(x1)];
                auto tmp10 = in_ptr6[static_cast<long>(x1)];
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = tmp0 ? tmp4 : tmp3;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                out_ptr2[static_cast<long>(x1 + (116L*x0))] = tmp12;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp9 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp30 = in_ptr4[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp31 = in_ptr5[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(1L + (2L*x0));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(116);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (232L*x2) + (45472L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 116L)))];
                            auto tmp8 = in_ptr2[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (232L*x2) + (45472L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 116L)))];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            return tmp9;
                        }
                        ;
                        auto tmp10 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp11 = tmp1 >= tmp4;
                        auto tmp12 = static_cast<long>(232);
                        auto tmp13 = tmp1 < tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr3[static_cast<long>((-22540L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp11 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp5 ? tmp10 : tmp16;
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp0 ? tmp18 : tmp17;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (232L*x2) + (45472L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 116L)))];
                            auto tmp22 = in_ptr2[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (232L*x2) + (45472L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 116L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp5 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-22540L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp11 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp5 ? tmp24 : tmp27;
                        auto tmp29 = tmp0 ? tmp18 : tmp28;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp33 = decltype(tmp29)(tmp29 * tmp32);
                        tmp_acc0 = tmp_acc0 + tmp19;
                        tmp_acc1 = tmp_acc1 + tmp33;
                    }
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr6[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (116L*x1) + (22736L*x0))];
                    auto tmp20 = in_ptr6[static_cast<long>(x2)];
                    auto tmp24 = in_ptr7[static_cast<long>(x2)];
                    auto tmp1 = c10::convert<long>(1L + (2L*x2));
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(116);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (232L*x1) + (45472L*x0) + (c10::div_floor_integer((1L + (2L*x2)), 116L)))];
                        auto tmp8 = in_ptr2[static_cast<long>((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (232L*x1) + (45472L*x0) + (c10::div_floor_integer((1L + (2L*x2)), 116L)))];
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        return tmp9;
                    }
                    ;
                    auto tmp10 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp11 = tmp1 >= tmp4;
                    auto tmp12 = static_cast<long>(232);
                    auto tmp13 = tmp1 < tmp12;
                    auto tmp14 = [&]
                    {
                        auto tmp15 = in_ptr3[static_cast<long>((-22540L) + x1 + (392L*x2) + (22736L*x0))];
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                    auto tmp17 = tmp5 ? tmp10 : tmp16;
                    auto tmp18 = static_cast<float>(0.0);
                    auto tmp19 = tmp0 ? tmp18 : tmp17;
                    auto tmp21 = static_cast<float>(1e-05);
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = 1 / std::sqrt(tmp22);
                    auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    out_ptr2[static_cast<long>(x2 + (116L*x1) + (22736L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp9 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr1;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (116L*x1) + (22736L*x0))];
                    auto tmp1 = c10::convert<long>(1L + (2L*x2));
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(116);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = c10::convert<long>((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x2)), 116L)));
                        auto tmp8 = static_cast<long>(0);
                        auto tmp9 = tmp7 >= tmp8;
                        auto tmp10 = static_cast<long>(116);
                        auto tmp11 = tmp7 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x2)), 116L)))) % static_cast<long>(116L))) + (232L*x1) + (45472L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x2)), 116L))), 116L)) % static_cast<long>(2L)))];
                            auto tmp14 = in_ptr2[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x2)), 116L)))) % static_cast<long>(116L))) + (232L*x1) + (45472L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x2)), 116L))), 116L)) % static_cast<long>(2L)))];
                            auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp17 = tmp7 >= tmp10;
                        auto tmp18 = static_cast<long>(232);
                        auto tmp19 = tmp7 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr3[static_cast<long>((-22736L) + x1 + (196L*(c10::div_floor_integer((1L + (2L*x2)), 116L))) + (392L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(116L))) + (22736L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp11 ? tmp16 : tmp22;
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp25 = tmp1 >= tmp4;
                    auto tmp26 = static_cast<long>(232);
                    auto tmp27 = tmp1 < tmp26;
                    auto tmp28 = [&]
                    {
                        auto tmp29 = in_ptr4[static_cast<long>((-22540L) + x1 + (392L*x2) + (22736L*x0))];
                        return tmp29;
                    }
                    ;
                    auto tmp30 = tmp25 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                    auto tmp31 = tmp5 ? tmp24 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    out_ptr0[static_cast<long>(x2 + (116L*x1) + (22736L*x0))] = tmp33;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr5[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr6[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                out_ptr2[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr2[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr7[static_cast<long>(x1)];
                auto tmp5 = in_ptr8[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp9 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(116);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<int>((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)));
                            auto tmp7 = static_cast<int>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<int>(116);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp12 = tmp10 & tmp4;
                            auto tmp11 = [&]
                            {
                                auto tmp13 = c10::convert<int>((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)));
                                auto tmp14 = static_cast<int>(0);
                                auto tmp15 = tmp13 >= tmp14;
                                auto tmp16 = static_cast<int>(116);
                                auto tmp17 = tmp13 < tmp16;
                                auto tmp19 = tmp17 & tmp12;
                                auto tmp18 = [&]
                                {
                                    auto tmp20 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)))) % static_cast<long>(116L))) + (232L*x2) + (232L*x2_inner) + (45472L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))), 116L)) % static_cast<long>(2L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                    auto tmp21 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)))) % static_cast<long>(116L))) + (232L*x2) + (232L*x2_inner) + (45472L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))), 116L)) % static_cast<long>(2L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                    auto tmp22 = tmp20 + tmp21;
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp19));
                                auto tmp24 = tmp13 >= tmp16;
                                auto tmp25 = static_cast<int>(232);
                                auto tmp26 = tmp13 < tmp25;
                                auto tmp28 = tmp24 & tmp12;
                                auto tmp27 = [&]
                                {
                                    auto tmp29 = masked_load(in_ptr2 + static_cast<long>((-22736L) + x2 + (196L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (22736L*x0)), to_float_mask(tmp28));
                                    return tmp29;
                                }
                                ;
                                auto tmp30 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp28));
                                auto tmp31 = to_float_mask(tmp17);
                                auto tmp32 = decltype(tmp23)::blendv(tmp30, tmp23, tmp31);
                                return tmp32;
                            }
                            ;
                            auto tmp33 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp12));
                            auto tmp34 = tmp6 >= tmp9;
                            auto tmp35 = static_cast<int>(232);
                            auto tmp36 = tmp6 < tmp35;
                            auto tmp38 = tmp34 & tmp4;
                            auto tmp37 = [&]
                            {
                                auto tmp39 = masked_load(in_ptr3 + static_cast<long>((-22736L) + x2 + (196L*(c10::div_floor_integer(x1, 116L))) + (392L*(static_cast<long>(x1) % static_cast<long>(116L))) + (22736L*x0)), to_float_mask(tmp38));
                                return tmp39;
                            }
                            ;
                            auto tmp40 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp38));
                            auto tmp41 = to_float_mask(tmp10);
                            auto tmp42 = decltype(tmp33)::blendv(tmp40, tmp33, tmp41);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp44 = tmp0 >= tmp3;
                        auto tmp45 = static_cast<int>(232);
                        auto tmp46 = tmp0 < tmp45;
                        auto tmp47 = [&]
                        {
                            auto tmp48 = masked_load(in_ptr4 + static_cast<long>((-22736L) + x2 + (196L*x1) + (22736L*x0)), to_float_mask(tmp44));
                            return tmp48;
                        }
                        ;
                        auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp44));
                        auto tmp50 = to_float_mask(tmp4);
                        auto tmp51 = decltype(tmp43)::blendv(tmp49, tmp43, tmp50);
                        tmp51.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (45472L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(116);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)));
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)));
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(116);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = in_ptr0[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)))) % static_cast<long>(116L))) + (232L*x2) + (45472L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))), 116L)) % static_cast<long>(2L)))];
                                    auto tmp19 = in_ptr1[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)))) % static_cast<long>(116L))) + (232L*x2) + (45472L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))), 116L)) % static_cast<long>(2L)))];
                                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp22 = tmp12 >= tmp15;
                                auto tmp23 = static_cast<long>(232);
                                auto tmp24 = tmp12 < tmp23;
                                auto tmp25 = [&]
                                {
                                    auto tmp26 = in_ptr2[static_cast<long>((-22736L) + x2 + (196L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (22736L*x0))];
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                                auto tmp28 = tmp16 ? tmp21 : tmp27;
                                return tmp28;
                            }
                            ;
                            auto tmp29 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp30 = tmp6 >= tmp9;
                            auto tmp31 = static_cast<long>(232);
                            auto tmp32 = tmp6 < tmp31;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr3[static_cast<long>((-22736L) + x2 + (196L*(c10::div_floor_integer(x1, 116L))) + (392L*(static_cast<long>(x1) % static_cast<long>(116L))) + (22736L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp30 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = tmp10 ? tmp29 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp38 = tmp0 >= tmp3;
                        auto tmp39 = static_cast<long>(232);
                        auto tmp40 = tmp0 < tmp39;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = in_ptr4[static_cast<long>((-22736L) + x2 + (196L*x1) + (22736L*x0))];
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp38 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp44 = tmp4 ? tmp37 : tmp43;
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (45472L*x0))] = tmp44;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                                float tmp1[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(196L + x2 + (392L*x0) + (45472L*x1)), static_cast<long>(392L), tmp1, 8);
                                at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(196L + x2 + (392L*x0) + (45472L*x1)), static_cast<long>(392L), tmp1, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0 + (116L*x2) + (116L*x2_inner) + (22736L*x1)));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (116L*x2) + (116L*x2_inner) + (22736L*x1)));
                                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                                    auto tmp3 = static_cast<float>(0.0);
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                    auto tmp8 = tmp6 - tmp7;
                                    auto tmp9 = tmp5 * tmp8;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                                }
                            }
                            for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0 + (116L*x2) + (22736L*x1)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(196L + x2 + (392L*x0) + (392L*x0_inner) + (45472L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (116L*x2) + (22736L*x1)));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                                auto tmp2 = static_cast<float>(0.0);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                                auto tmp7 = tmp5 - tmp6;
                                auto tmp8 = tmp4 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp4;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
                #pragma GCC ivdep
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                        {
                            for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr5[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                                auto tmp1 = out_ptr0[static_cast<long>(196L + x2 + (392L*x0) + (45472L*x1))];
                                auto tmp4 = in_ptr6[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                                auto tmp5 = in_ptr7[static_cast<long>(x0)];
                                auto tmp2 = static_cast<float>(0.0);
                                auto tmp3 = tmp0 ? tmp2 : tmp1;
                                auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                                auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                                tmp_acc0 = tmp_acc0 + tmp3;
                                tmp_acc1 = tmp_acc1 + tmp7;
                            }
                        }
                        out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                        out_ptr2[static_cast<long>(x0)] = tmp_acc1;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>(x0)];
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(196L + x1 + (392L*x2) + (45472L*x0)), static_cast<long>(392L), tmp1, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                                auto tmp3 = static_cast<float>(0.0);
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp12 = tmp10 * tmp11;
                                auto tmp13 = tmp5 * tmp12;
                                tmp13.store(out_ptr3 + static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(112L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = flag_to_float_scalar(in_ptr5[static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0))]); return flag_to_float_vec(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(196L + x1 + (392L*x2) + (45472L*x0)));
                            auto tmp5 = in_ptr8[static_cast<long>(x2)];
                            auto tmp9 = in_ptr9[static_cast<long>(x2)];
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp12.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x2 + (116L*x1) + (22736L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(196L + x1 + (392L*x2) + (45472L*x0))];
                            auto tmp4 = in_ptr8[static_cast<long>(x2)];
                            auto tmp8 = in_ptr9[static_cast<long>(x2)];
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = tmp0 ? tmp2 : tmp1;
                            auto tmp5 = static_cast<float>(1e-05);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = 1 / std::sqrt(tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                            out_ptr3[static_cast<long>(x2 + (116L*x1) + (22736L*x0))] = tmp10;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp9 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp26 = in_ptr3[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp27 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(1L + (2L*x0));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(116);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer((1L + (2L*x0)), 116L))) + (392L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (45472L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(232);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-22540L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp0 ? tmp16 : tmp15;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer((1L + (2L*x0)), 116L))) + (392L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (45472L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = [&]
                        {
                            auto tmp22 = in_ptr2[static_cast<long>((-22540L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp24 = tmp5 ? tmp20 : tmp23;
                        auto tmp25 = tmp0 ? tmp16 : tmp24;
                        auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                        auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp29;
                    }
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x2) + (22736L*x0))];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp22 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(1L + (2L*x1));
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(116);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer((1L + (2L*x1)), 116L))) + (392L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(116L))) + (45472L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(232);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-22540L) + x2 + (392L*x1) + (22736L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp0 ? tmp16 : tmp15;
                    auto tmp19 = static_cast<float>(1e-05);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = 1 / std::sqrt(tmp20);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp17)(tmp17 * tmp23);
                    out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp9 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x2) + (22736L*x0))];
                    auto tmp32 = in_ptr4[static_cast<long>(x1)];
                    auto tmp36 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(1L + (2L*x1));
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(116);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = c10::convert<long>((2L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x1)), 116L)));
                        auto tmp8 = static_cast<long>(0);
                        auto tmp9 = tmp7 >= tmp8;
                        auto tmp10 = static_cast<long>(116);
                        auto tmp11 = tmp7 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>(x2 + (196L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x1)), 116L))), 116L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(((2L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(116L))) + (c10::div_floor_integer((1L + (2L*x1)), 116L)))) % static_cast<long>(116L))) + (45472L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp7 >= tmp10;
                        auto tmp16 = static_cast<long>(232);
                        auto tmp17 = tmp7 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-22736L) + x2 + (196L*(c10::div_floor_integer((1L + (2L*x1)), 116L))) + (392L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(116L))) + (22736L*x0))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp23 = tmp1 >= tmp4;
                    auto tmp24 = static_cast<long>(232);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr3[static_cast<long>((-22540L) + x2 + (392L*x1) + (22736L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp5 ? tmp22 : tmp28;
                    auto tmp30 = static_cast<float>(0.0);
                    auto tmp31 = tmp0 ? tmp30 : tmp29;
                    auto tmp33 = static_cast<float>(1e-05);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    auto tmp35 = 1 / std::sqrt(tmp34);
                    auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                    auto tmp38 = decltype(tmp31)(tmp31 * tmp37);
                    out_ptr0[static_cast<long>(x2 + (196L*x1) + (22736L*x0))] = tmp31;
                    out_ptr1[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp38;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (22736L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (22736L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (116L*x2) + (116L*x2_inner) + (22736L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (22736L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (116L*x2) + (22736L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
        #pragma GCC ivdep
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x2 + (196L*x0) + (22736L*x1))];
                        auto tmp1 = in_ptr6[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp2 = in_ptr7[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        tmp_acc0 = tmp_acc0 + tmp0;
                        tmp_acc1 = tmp_acc1 + tmp4;
                    }
                }
                out_ptr2[static_cast<long>(x0)] = tmp_acc0;
                out_ptr3[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr3[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp9 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(232L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(116);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<int>((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)));
                            auto tmp7 = static_cast<int>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<int>(116);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp12 = tmp10 & tmp4;
                            auto tmp11 = [&]
                            {
                                auto tmp13 = c10::convert<int>((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)));
                                auto tmp14 = static_cast<int>(0);
                                auto tmp15 = tmp13 >= tmp14;
                                auto tmp16 = static_cast<int>(116);
                                auto tmp17 = tmp13 < tmp16;
                                auto tmp19 = tmp17 & tmp12;
                                auto tmp18 = [&]
                                {
                                    auto tmp20 = masked_load(in_ptr0 + static_cast<long>(x2 + (196L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))), 116L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)))) % static_cast<long>(116L))) + (45472L*x0)), to_float_mask(tmp19));
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp19));
                                auto tmp22 = tmp13 >= tmp16;
                                auto tmp23 = static_cast<int>(232);
                                auto tmp24 = tmp13 < tmp23;
                                auto tmp26 = tmp22 & tmp12;
                                auto tmp25 = [&]
                                {
                                    auto tmp27 = masked_load(in_ptr1 + static_cast<long>((-22736L) + x2 + (196L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (22736L*x0)), to_float_mask(tmp26));
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp26));
                                auto tmp29 = to_float_mask(tmp17);
                                auto tmp30 = decltype(tmp21)::blendv(tmp28, tmp21, tmp29);
                                return tmp30;
                            }
                            ;
                            auto tmp31 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp12));
                            auto tmp32 = tmp6 >= tmp9;
                            auto tmp33 = static_cast<int>(232);
                            auto tmp34 = tmp6 < tmp33;
                            auto tmp36 = tmp32 & tmp4;
                            auto tmp35 = [&]
                            {
                                auto tmp37 = masked_load(in_ptr2 + static_cast<long>((-22736L) + x2 + (196L*(c10::div_floor_integer(x1, 116L))) + (392L*(static_cast<long>(x1) % static_cast<long>(116L))) + (22736L*x0)), to_float_mask(tmp36));
                                return tmp37;
                            }
                            ;
                            auto tmp38 = decltype(tmp35())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp35(), to_float_mask(tmp36));
                            auto tmp39 = to_float_mask(tmp10);
                            auto tmp40 = decltype(tmp31)::blendv(tmp38, tmp31, tmp39);
                            return tmp40;
                        }
                        ;
                        auto tmp41 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp42 = tmp0 >= tmp3;
                        auto tmp43 = static_cast<int>(232);
                        auto tmp44 = tmp0 < tmp43;
                        auto tmp45 = [&]
                        {
                            auto tmp46 = masked_load(in_ptr3 + static_cast<long>((-22736L) + x2 + (196L*x1) + (22736L*x0)), to_float_mask(tmp42));
                            return tmp46;
                        }
                        ;
                        auto tmp47 = decltype(tmp45())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp45(), to_float_mask(tmp42));
                        auto tmp48 = to_float_mask(tmp4);
                        auto tmp49 = decltype(tmp41)::blendv(tmp47, tmp41, tmp48);
                        tmp49.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (45472L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(116);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<long>((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)));
                            auto tmp7 = static_cast<long>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<long>(116);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)));
                                auto tmp13 = static_cast<long>(0);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = static_cast<long>(116);
                                auto tmp16 = tmp12 < tmp15;
                                auto tmp17 = [&]
                                {
                                    auto tmp18 = in_ptr0[static_cast<long>(x2 + (196L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))), 116L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L)))) % static_cast<long>(116L))) + (45472L*x0))];
                                    return tmp18;
                                }
                                ;
                                auto tmp19 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                                auto tmp20 = tmp12 >= tmp15;
                                auto tmp21 = static_cast<long>(232);
                                auto tmp22 = tmp12 < tmp21;
                                auto tmp23 = [&]
                                {
                                    auto tmp24 = in_ptr1[static_cast<long>((-22736L) + x2 + (196L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L))), 116L)) % static_cast<long>(2L))) + (392L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(116L))) + (c10::div_floor_integer(x1, 116L)))) % static_cast<long>(116L))) + (22736L*x0))];
                                    return tmp24;
                                }
                                ;
                                auto tmp25 = tmp20 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                                auto tmp26 = tmp16 ? tmp19 : tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp28 = tmp6 >= tmp9;
                            auto tmp29 = static_cast<long>(232);
                            auto tmp30 = tmp6 < tmp29;
                            auto tmp31 = [&]
                            {
                                auto tmp32 = in_ptr2[static_cast<long>((-22736L) + x2 + (196L*(c10::div_floor_integer(x1, 116L))) + (392L*(static_cast<long>(x1) % static_cast<long>(116L))) + (22736L*x0))];
                                return tmp32;
                            }
                            ;
                            auto tmp33 = tmp28 ? tmp31() : static_cast<decltype(tmp31())>(0.0);
                            auto tmp34 = tmp10 ? tmp27 : tmp33;
                            return tmp34;
                        }
                        ;
                        auto tmp35 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp36 = tmp0 >= tmp3;
                        auto tmp37 = static_cast<long>(232);
                        auto tmp38 = tmp0 < tmp37;
                        auto tmp39 = [&]
                        {
                            auto tmp40 = in_ptr3[static_cast<long>((-22736L) + x2 + (196L*x1) + (22736L*x0))];
                            return tmp40;
                        }
                        ;
                        auto tmp41 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                        auto tmp42 = tmp4 ? tmp35 : tmp41;
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (45472L*x0))] = tmp42;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                                float tmp1[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(196L + x2 + (392L*x0) + (45472L*x1)), static_cast<long>(392L), tmp1, 8);
                                at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(196L + x2 + (392L*x0) + (45472L*x1)), static_cast<long>(392L), tmp1, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x0 + (116L*x2) + (116L*x2_inner) + (22736L*x1)));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (116L*x2) + (116L*x2_inner) + (22736L*x1)));
                                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                    auto tmp3 = static_cast<float>(0.0);
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                    auto tmp8 = tmp6 - tmp7;
                                    auto tmp9 = tmp5 * tmp8;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                                }
                            }
                            for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x0 + (116L*x2) + (22736L*x1)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(196L + x2 + (392L*x0) + (392L*x0_inner) + (45472L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (116L*x2) + (22736L*x1)));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp2 = static_cast<float>(0.0);
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                                auto tmp7 = tmp5 - tmp6;
                                auto tmp8 = tmp4 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp4;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
                #pragma GCC ivdep
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                        {
                            for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr4[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                                auto tmp1 = out_ptr0[static_cast<long>(196L + x2 + (392L*x0) + (45472L*x1))];
                                auto tmp4 = in_ptr5[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                                auto tmp5 = in_ptr6[static_cast<long>(x0)];
                                auto tmp2 = static_cast<float>(0.0);
                                auto tmp3 = tmp0 ? tmp2 : tmp1;
                                auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                                auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                                tmp_acc0 = tmp_acc0 + tmp3;
                                tmp_acc1 = tmp_acc1 + tmp7;
                            }
                        }
                        out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                        out_ptr2[static_cast<long>(x0)] = tmp_acc1;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>(x0)];
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(196L + x1 + (392L*x2) + (45472L*x0)), static_cast<long>(392L), tmp1, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                                auto tmp3 = static_cast<float>(0.0);
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp12 = tmp10 * tmp11;
                                auto tmp13 = tmp5 * tmp12;
                                tmp13.store(out_ptr3 + static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(112L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = flag_to_float_scalar(in_ptr4[static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0))]); return flag_to_float_vec(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(196L + x1 + (392L*x2) + (45472L*x0)));
                            auto tmp5 = in_ptr7[static_cast<long>(x2)];
                            auto tmp9 = in_ptr8[static_cast<long>(x2)];
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp12.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (116L*x1) + (116L*x1_inner) + (22736L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(116L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x2 + (116L*x1) + (22736L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(196L + x1 + (392L*x2) + (45472L*x0))];
                            auto tmp4 = in_ptr7[static_cast<long>(x2)];
                            auto tmp8 = in_ptr8[static_cast<long>(x2)];
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = tmp0 ? tmp2 : tmp1;
                            auto tmp5 = static_cast<float>(1e-05);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = 1 / std::sqrt(tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                            out_ptr3[static_cast<long>(x2 + (116L*x1) + (22736L*x0))] = tmp10;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp9 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp26 = in_ptr3[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp27 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(1L + (2L*x0));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(116);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer((1L + (2L*x0)), 116L))) + (392L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (45472L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(232);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-22540L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp0 ? tmp16 : tmp15;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer((1L + (2L*x0)), 116L))) + (392L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(116L))) + (45472L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = [&]
                        {
                            auto tmp22 = in_ptr2[static_cast<long>((-22540L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp24 = tmp5 ? tmp20 : tmp23;
                        auto tmp25 = tmp0 ? tmp16 : tmp24;
                        auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                        auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp29;
                    }
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x2) + (22736L*x0))];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp22 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(1L + (2L*x1));
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(116);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer((1L + (2L*x1)), 116L))) + (392L*(static_cast<long>((1L + (2L*x1))) % static_cast<long>(116L))) + (45472L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(232);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-22540L) + x2 + (392L*x1) + (22736L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp0 ? tmp16 : tmp15;
                    auto tmp19 = static_cast<float>(1e-05);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = 1 / std::sqrt(tmp20);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp17)(tmp17 * tmp23);
                    out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (116L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                        auto tmp5 = in_ptr2[static_cast<long>(x0 + (116L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp4 = tmp2 ? tmp1 : tmp3;
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                        tmp_acc0 = tmp_acc0 + tmp4;
                        tmp_acc1 = tmp_acc1 + tmp8;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (116L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x0))];
                    auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp9 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                    in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp11;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp26 = in_ptr3[static_cast<long>(x0 + (116L*x2) + (22736L*x1))];
                        auto tmp27 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(2L*x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(116);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer(x0, 58L))) + (392L*(static_cast<long>((2L*x0)) % static_cast<long>(116L))) + (45472L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(232);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-22736L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp0 ? tmp16 : tmp15;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer(x0, 58L))) + (392L*(static_cast<long>((2L*x0)) % static_cast<long>(116L))) + (45472L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = [&]
                        {
                            auto tmp22 = in_ptr2[static_cast<long>((-22736L) + x2 + (392L*x0) + (22736L*x1))];
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp24 = tmp5 ? tmp20 : tmp23;
                        auto tmp25 = tmp0 ? tmp16 : tmp24;
                        auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                        auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp29;
                    }
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (116L*x2) + (22736L*x0))];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp22 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(2L*x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(116);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x2 + (196L*(c10::div_floor_integer(x1, 58L))) + (392L*(static_cast<long>((2L*x1)) % static_cast<long>(116L))) + (45472L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(232);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-22736L) + x2 + (392L*x1) + (22736L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp0 ? tmp16 : tmp15;
                    auto tmp19 = static_cast<float>(1e-05);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = 1 / std::sqrt(tmp20);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp17)(tmp17 * tmp23);
                    out_ptr2[static_cast<long>(x1 + (116L*x2) + (22736L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (116L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (116L*x1)));
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
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (116L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (116L*x1))];
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(112L); x0<static_cast<long>(116L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp4 = 1 / std::sqrt(tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            in_out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (116L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(112L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (116L*x0))];
                auto tmp1 = in_ptr3[static_cast<long>(x1)];
                auto tmp5 = in_ptr4[static_cast<long>(x1)];
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp4 = 1 / std::sqrt(tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                in_out_ptr1[static_cast<long>(x1 + (116L*x0))] = tmp7;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(1L + (2L*x0) + (2L*x0_inner) + (116L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>(1L + (2L*x0) + (2L*x0_inner) + (116L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(1L + (2L*x0) + (116L*x1))];
                        auto tmp2 = in_ptr2[static_cast<long>(1L + (2L*x0) + (116L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0 + (58L*x1))];
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                        tmp_acc0 = tmp_acc0 + tmp5;
                        tmp_acc1 = tmp_acc1 + tmp9;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(1L + (2L*x1) + (2L*x1_inner) + (116L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(1L + (2L*x1) + (2L*x1_inner) + (116L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    tmp14.store(out_ptr2 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(1L + (2L*x1) + (116L*x0))];
                    auto tmp2 = in_ptr2[static_cast<long>(1L + (2L*x1) + (116L*x0))];
                    auto tmp6 = in_ptr5[static_cast<long>(x1)];
                    auto tmp10 = in_ptr6[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp0 ? tmp4 : tmp3;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    out_ptr2[static_cast<long>(x1 + (58L*x0))] = tmp12;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        tmp_acc0 = tmp_acc0 + tmp0;
                        tmp_acc1 = tmp_acc1 + tmp4;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr3[static_cast<long>(x1)];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp5 = in_ptr2[static_cast<long>(x0 + (58L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp4 = tmp2 ? tmp1 : tmp3;
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                        tmp_acc0 = tmp_acc0 + tmp4;
                        tmp_acc1 = tmp_acc1 + tmp8;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp9 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp11;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x2) + (45472L*x1))];
                            auto tmp30 = in_ptr4[static_cast<long>(x0 + (58L*x2) + (45472L*x1))];
                            auto tmp31 = in_ptr5[static_cast<long>(x0)];
                            auto tmp1 = c10::convert<long>(1L + (2L*x0));
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 >= tmp2;
                            auto tmp4 = static_cast<long>(58);
                            auto tmp5 = tmp1 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(58L))) + (116L*x2) + (90944L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 58L)))];
                                auto tmp8 = in_ptr2[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(58L))) + (116L*x2) + (90944L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 58L)))];
                                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                return tmp9;
                            }
                            ;
                            auto tmp10 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp11 = tmp1 >= tmp4;
                            auto tmp12 = static_cast<long>(116);
                            auto tmp13 = tmp1 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr3[static_cast<long>((-44688L) + x2 + (1568L*x0) + (45472L*x1))];
                                return tmp15;
                            }
                            ;
                            auto tmp16 = tmp11 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp17 = tmp5 ? tmp10 : tmp16;
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp0 ? tmp18 : tmp17;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(58L))) + (116L*x2) + (90944L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 58L)))];
                                auto tmp22 = in_ptr2[static_cast<long>((2L*(static_cast<long>((1L + (2L*x0))) % static_cast<long>(58L))) + (116L*x2) + (90944L*x1) + (c10::div_floor_integer((1L + (2L*x0)), 58L)))];
                                auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp5 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp25 = [&]
                            {
                                auto tmp26 = in_ptr3[static_cast<long>((-44688L) + x2 + (1568L*x0) + (45472L*x1))];
                                return tmp26;
                            }
                            ;
                            auto tmp27 = tmp11 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                            auto tmp28 = tmp5 ? tmp24 : tmp27;
                            auto tmp29 = tmp0 ? tmp18 : tmp28;
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = decltype(tmp29)(tmp29 * tmp32);
                            tmp_acc0 = tmp_acc0 + tmp19;
                            tmp_acc1 = tmp_acc1 + tmp33;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(58L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (58L*x1) + (45472L*x0))];
                        auto tmp20 = in_ptr6[static_cast<long>(x2)];
                        auto tmp24 = in_ptr7[static_cast<long>(x2)];
                        auto tmp1 = c10::convert<long>(1L + (2L*x2));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(58);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (116L*x1) + (90944L*x0) + (c10::div_floor_integer((1L + (2L*x2)), 58L)))];
                            auto tmp8 = in_ptr2[static_cast<long>((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (116L*x1) + (90944L*x0) + (c10::div_floor_integer((1L + (2L*x2)), 58L)))];
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            return tmp9;
                        }
                        ;
                        auto tmp10 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp11 = tmp1 >= tmp4;
                        auto tmp12 = static_cast<long>(116);
                        auto tmp13 = tmp1 < tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr3[static_cast<long>((-44688L) + x1 + (1568L*x2) + (45472L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp11 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp5 ? tmp10 : tmp16;
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp0 ? tmp18 : tmp17;
                        auto tmp21 = static_cast<float>(1e-05);
                        auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                        auto tmp23 = 1 / std::sqrt(tmp22);
                        auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
                        auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                        out_ptr2[static_cast<long>(x2 + (58L*x1) + (45472L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        tmp_acc0 = tmp_acc0 + tmp0;
                        tmp_acc1 = tmp_acc1 + tmp4;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr3[static_cast<long>(x1)];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp5 = in_ptr2[static_cast<long>(x0 + (58L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp4 = tmp2 ? tmp1 : tmp3;
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                        tmp_acc0 = tmp_acc0 + tmp4;
                        tmp_acc1 = tmp_acc1 + tmp8;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp9 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp11;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(58L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (58L*x1) + (45472L*x0))];
                        auto tmp1 = c10::convert<long>(1L + (2L*x2));
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(58);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (c10::div_floor_integer((1L + (2L*x2)), 58L)));
                            auto tmp8 = static_cast<long>(0);
                            auto tmp9 = tmp7 >= tmp8;
                            auto tmp10 = static_cast<long>(58);
                            auto tmp11 = tmp7 < tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = in_ptr1[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (c10::div_floor_integer((1L + (2L*x2)), 58L)))) % static_cast<long>(58L))) + (116L*x1) + (90944L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (c10::div_floor_integer((1L + (2L*x2)), 58L))), 58L)) % static_cast<long>(2L)))];
                                auto tmp14 = in_ptr2[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (c10::div_floor_integer((1L + (2L*x2)), 58L)))) % static_cast<long>(58L))) + (116L*x1) + (90944L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (c10::div_floor_integer((1L + (2L*x2)), 58L))), 58L)) % static_cast<long>(2L)))];
                                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                                return tmp15;
                            }
                            ;
                            auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                            auto tmp17 = tmp7 >= tmp10;
                            auto tmp18 = static_cast<long>(116);
                            auto tmp19 = tmp7 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr3[static_cast<long>((-45472L) + x1 + (784L*(c10::div_floor_integer((1L + (2L*x2)), 58L))) + (1568L*(static_cast<long>((1L + (2L*x2))) % static_cast<long>(58L))) + (45472L*x0))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp23 = tmp11 ? tmp16 : tmp22;
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp25 = tmp1 >= tmp4;
                        auto tmp26 = static_cast<long>(116);
                        auto tmp27 = tmp1 < tmp26;
                        auto tmp28 = [&]
                        {
                            auto tmp29 = in_ptr4[static_cast<long>((-44688L) + x1 + (1568L*x2) + (45472L*x0))];
                            return tmp29;
                        }
                        ;
                        auto tmp30 = tmp25 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                        auto tmp31 = tmp5 ? tmp24 : tmp30;
                        auto tmp32 = static_cast<float>(0.0);
                        auto tmp33 = tmp0 ? tmp32 : tmp31;
                        out_ptr0[static_cast<long>(x2 + (58L*x1) + (45472L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp1 = in_ptr5[static_cast<long>(x0 + (58L*x1))];
                        auto tmp2 = in_ptr6[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        tmp_acc0 = tmp_acc0 + tmp0;
                        tmp_acc1 = tmp_acc1 + tmp4;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr2[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>(x0)];
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr7[static_cast<long>(x1)];
                    auto tmp5 = in_ptr8[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        tmp_acc0 = tmp_acc0 + tmp0;
                        tmp_acc1 = tmp_acc1 + tmp4;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr3[static_cast<long>(x1)];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp5 = in_ptr2[static_cast<long>(x0 + (58L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp4 = tmp2 ? tmp1 : tmp3;
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                        tmp_acc0 = tmp_acc0 + tmp4;
                        tmp_acc1 = tmp_acc1 + tmp8;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp9 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp11;
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(116L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(58);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<int>((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L)));
                            auto tmp7 = static_cast<int>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<int>(58);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp12 = tmp10 & tmp4;
                            auto tmp11 = [&]
                            {
                                auto tmp13 = c10::convert<int>((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L)))) % static_cast<long>(58L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L))), 58L)) % static_cast<long>(2L)));
                                auto tmp14 = static_cast<int>(0);
                                auto tmp15 = tmp13 >= tmp14;
                                auto tmp16 = static_cast<int>(58);
                                auto tmp17 = tmp13 < tmp16;
                                auto tmp19 = tmp17 & tmp12;
                                auto tmp18 = [&]
                                {
                                    auto tmp20 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L)))) % static_cast<long>(58L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L))), 58L)) % static_cast<long>(2L)))) % static_cast<long>(58L))) + (116L*x2) + (116L*x2_inner) + (90944L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L)))) % static_cast<long>(58L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L))), 58L)) % static_cast<long>(2L))), 58L)) % static_cast<long>(2L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                    auto tmp21 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((2L*(static_cast<long>(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L)))) % static_cast<long>(58L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L))), 58L)) % static_cast<long>(2L)))) % static_cast<long>(58L))) + (116L*x2) + (116L*x2_inner) + (90944L*x0) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L)))) % static_cast<long>(58L))) + (static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L))), 58L)) % static_cast<long>(2L))), 58L)) % static_cast<long>(2L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                    auto tmp22 = tmp20 + tmp21;
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp19));
                                auto tmp24 = tmp13 >= tmp16;
                                auto tmp25 = static_cast<int>(116);
                                auto tmp26 = tmp13 < tmp25;
                                auto tmp28 = tmp24 & tmp12;
                                auto tmp27 = [&]
                                {
                                    auto tmp29 = masked_load(in_ptr2 + static_cast<long>((-45472L) + x2 + (784L*(static_cast<long>(c10::div_floor_integer(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L))), 58L)) % static_cast<long>(2L))) + (1568L*(static_cast<long>(((2L*(static_cast<long>(x1) % static_cast<long>(58L))) + (c10::div_floor_integer(x1, 58L)))) % static_cast<long>(58L))) + (45472L*x0)), to_float_mask(tmp28));
                                    return tmp29;
                                }
                                ;
                                auto tmp30 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp28));
                                auto tmp31 = to_float_mask(tmp17);
                                auto tmp32 = decltype(tmp23)::blendv(tmp30, tmp23, tmp31);
                                return tmp32;
                            }
                            ;
                            auto tmp33 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp12));
                            auto tmp34 = tmp6 >= tmp9;
                            auto tmp35 = static_cast<int>(116);
                            auto tmp36 = tmp6 < tmp35;
                            auto tmp38 = tmp34 & tmp4;
                            auto tmp37 = [&]
                            {
                                auto tmp39 = masked_load(in_ptr3 + static_cast<long>((-45472L) + x2 + (784L*(c10::div_floor_integer(x1, 58L))) + (1568L*(static_cast<long>(x1) % static_cast<long>(58L))) + (45472L*x0)), to_float_mask(tmp38));
                                return tmp39;
                            }
                            ;
                            auto tmp40 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp38));
                            auto tmp41 = to_float_mask(tmp10);
                            auto tmp42 = decltype(tmp33)::blendv(tmp40, tmp33, tmp41);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp44 = tmp0 >= tmp3;
                        auto tmp45 = static_cast<int>(116);
                        auto tmp46 = tmp0 < tmp45;
                        auto tmp47 = [&]
                        {
                            auto tmp48 = masked_load(in_ptr4 + static_cast<long>((-45472L) + x2 + (784L*x1) + (45472L*x0)), to_float_mask(tmp44));
                            return tmp48;
                        }
                        ;
                        auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp44));
                        auto tmp50 = to_float_mask(tmp4);
                        auto tmp51 = decltype(tmp43)::blendv(tmp49, tmp43, tmp50);
                        tmp51.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (90944L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(784L + x2 + (1568L*x0) + (90944L*x1)), static_cast<long>(1568L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(784L + x2 + (1568L*x0) + (90944L*x1)), static_cast<long>(1568L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0 + (58L*x2) + (58L*x2_inner) + (45472L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (58L*x2) + (58L*x2_inner) + (45472L*x1)));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                                auto tmp3 = static_cast<float>(0.0);
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x0 + (58L*x2) + (45472L*x1))];
                            auto tmp1 = out_ptr0[static_cast<long>(784L + x2 + (1568L*x0) + (90944L*x1))];
                            auto tmp4 = in_ptr6[static_cast<long>(x0 + (58L*x2) + (45472L*x1))];
                            auto tmp5 = in_ptr7[static_cast<long>(x0)];
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = tmp0 ? tmp2 : tmp1;
                            auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp3;
                            tmp_acc1 = tmp_acc1 + tmp7;
                        }
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr2[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>(x0)];
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(784L + x1 + (1568L*x2) + (90944L*x0)), static_cast<long>(1568L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = tmp5 * tmp12;
                            tmp13.store(out_ptr3 + static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(56L); x2<static_cast<long>(58L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = flag_to_float_scalar(in_ptr5[static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0))]); return flag_to_float_vec(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(784L + x1 + (1568L*x2) + (90944L*x0)));
                        auto tmp5 = in_ptr8[static_cast<long>(x2)];
                        auto tmp9 = in_ptr9[static_cast<long>(x2)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp4 * tmp11;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp12.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        tmp_acc0 = tmp_acc0 + tmp0;
                        tmp_acc1 = tmp_acc1 + tmp4;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp1 = in_ptr3[static_cast<long>(x1)];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (58L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                        auto tmp5 = in_ptr2[static_cast<long>(x0 + (58L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp4 = tmp2 ? tmp1 : tmp3;
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                        tmp_acc0 = tmp_acc0 + tmp4;
                        tmp_acc1 = tmp_acc1 + tmp8;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (58L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                    auto tmp3 = in_out_ptr1[static_cast<long>(x1 + (58L*x0))];
                    auto tmp5 = in_ptr4[static_cast<long>(x1)];
                    auto tmp9 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                    in_out_ptr1[static_cast<long>(x1 + (58L*x0))] = tmp11;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1568L*x0) + (90944L*x1)), static_cast<long>(1568L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1568L*x0) + (90944L*x1)), static_cast<long>(1568L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (58L*x2) + (58L*x2_inner) + (45472L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (58L*x2) + (58L*x2_inner) + (45472L*x1)));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = static_cast<float>(0.0);
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x2) + (45472L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2 + (1568L*x0) + (90944L*x1))];
                            auto tmp4 = in_ptr2[static_cast<long>(x0 + (58L*x2) + (45472L*x1))];
                            auto tmp5 = in_ptr3[static_cast<long>(x0)];
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = tmp0 ? tmp2 : tmp1;
                            auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp3;
                            tmp_acc1 = tmp_acc1 + tmp7;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = 1 / std::sqrt(tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    in_out_ptr0[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1568L*x2) + (90944L*x0)), static_cast<long>(1568L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = tmp5 * tmp12;
                            tmp13.store(out_ptr2 + static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(56L); x2<static_cast<long>(58L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = flag_to_float_scalar(in_ptr0[static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0))]); return flag_to_float_vec(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1568L*x2) + (90944L*x0)));
                        auto tmp5 = in_ptr4[static_cast<long>(x2)];
                        auto tmp9 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp4 * tmp11;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp12.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (58L*x1) + (58L*x1_inner) + (45472L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(301056L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, primals_339, convolution, relu, getitem, getitem_1, convolution_1, add_3, convolution_2, convolution_3, relu_2, convolution_4, add_9, convolution_5, getitem_3, convolution_6, relu_4, convolution_7, add_15, convolution_8, getitem_5, convolution_9, relu_6, convolution_10, add_21, convolution_11, getitem_7, convolution_12, relu_8, convolution_13, add_27, convolution_14, view_7, convolution_15, add_31, convolution_16, convolution_17, relu_11, convolution_18, add_37, convolution_19, getitem_9, convolution_20, relu_13, convolution_21, add_43, convolution_22, getitem_11, convolution_23, relu_15, convolution_24, add_49, convolution_25, getitem_13, convolution_26, relu_17, convolution_27, add_55, convolution_28, getitem_15, convolution_29, relu_19, convolution_30, add_61, convolution_31, getitem_17, convolution_32, relu_21, convolution_33, add_67, convolution_34, getitem_19, convolution_35, relu_23, convolution_36, add_73, convolution_37, getitem_21, convolution_38, relu_25, convolution_39, add_79, convolution_40, view_23, convolution_41, add_83, convolution_42, convolution_43, relu_28, convolution_44, add_89, convolution_45, getitem_23, convolution_46, relu_30, convolution_47, add_95, convolution_48, getitem_25, convolution_49, relu_32, convolution_50, add_101, convolution_51, getitem_27, convolution_52, relu_34, convolution_53, add_107, convolution_54, view_31, convolution_55, mean, permute_17, le, le_1, le_3, le_5, le_7, le_9, le_10, le_12, le_14, le_16, le_18, le_20, le_22, le_24, le_26, le_27, le_29, le_31, le_33, le_35, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (24, ), (1, ))
    assert_size_stride(primals_4, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_7, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_8, (58, ), (1, ))
    assert_size_stride(primals_10, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_11, (58, ), (1, ))
    assert_size_stride(primals_13, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (58, ), (1, ))
    assert_size_stride(primals_16, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_17, (58, ), (1, ))
    assert_size_stride(primals_19, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_20, (58, ), (1, ))
    assert_size_stride(primals_22, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (58, ), (1, ))
    assert_size_stride(primals_25, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_26, (58, ), (1, ))
    assert_size_stride(primals_28, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_29, (58, ), (1, ))
    assert_size_stride(primals_31, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (58, ), (1, ))
    assert_size_stride(primals_34, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_35, (58, ), (1, ))
    assert_size_stride(primals_37, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_38, (58, ), (1, ))
    assert_size_stride(primals_40, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_41, (58, ), (1, ))
    assert_size_stride(primals_43, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_44, (58, ), (1, ))
    assert_size_stride(primals_46, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (116, ), (1, ))
    assert_size_stride(primals_49, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_50, (116, ), (1, ))
    assert_size_stride(primals_52, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_53, (116, ), (1, ))
    assert_size_stride(primals_55, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (116, ), (1, ))
    assert_size_stride(primals_58, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_59, (116, ), (1, ))
    assert_size_stride(primals_61, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_62, (116, ), (1, ))
    assert_size_stride(primals_64, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_65, (116, ), (1, ))
    assert_size_stride(primals_67, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_68, (116, ), (1, ))
    assert_size_stride(primals_70, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_71, (116, ), (1, ))
    assert_size_stride(primals_73, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (116, ), (1, ))
    assert_size_stride(primals_76, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_77, (116, ), (1, ))
    assert_size_stride(primals_79, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_80, (116, ), (1, ))
    assert_size_stride(primals_82, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_83, (116, ), (1, ))
    assert_size_stride(primals_85, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_86, (116, ), (1, ))
    assert_size_stride(primals_88, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_89, (116, ), (1, ))
    assert_size_stride(primals_91, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (116, ), (1, ))
    assert_size_stride(primals_94, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_95, (116, ), (1, ))
    assert_size_stride(primals_97, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_98, (116, ), (1, ))
    assert_size_stride(primals_100, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (116, ), (1, ))
    assert_size_stride(primals_103, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_104, (116, ), (1, ))
    assert_size_stride(primals_106, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_107, (116, ), (1, ))
    assert_size_stride(primals_109, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (116, ), (1, ))
    assert_size_stride(primals_112, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_113, (116, ), (1, ))
    assert_size_stride(primals_115, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_116, (116, ), (1, ))
    assert_size_stride(primals_118, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (116, ), (1, ))
    assert_size_stride(primals_121, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_122, (116, ), (1, ))
    assert_size_stride(primals_124, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (232, ), (1, ))
    assert_size_stride(primals_127, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_128, (232, ), (1, ))
    assert_size_stride(primals_130, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_131, (232, ), (1, ))
    assert_size_stride(primals_133, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (232, ), (1, ))
    assert_size_stride(primals_136, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_137, (232, ), (1, ))
    assert_size_stride(primals_139, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_140, (232, ), (1, ))
    assert_size_stride(primals_142, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (232, ), (1, ))
    assert_size_stride(primals_145, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_146, (232, ), (1, ))
    assert_size_stride(primals_148, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_149, (232, ), (1, ))
    assert_size_stride(primals_151, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (232, ), (1, ))
    assert_size_stride(primals_154, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_155, (232, ), (1, ))
    assert_size_stride(primals_157, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_158, (232, ), (1, ))
    assert_size_stride(primals_160, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (232, ), (1, ))
    assert_size_stride(primals_163, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_164, (232, ), (1, ))
    assert_size_stride(primals_166, (1024, 464, 1, 1), (464, 1, 1, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_171, (24, ), (1, ))
    assert_size_stride(primals_172, (24, ), (1, ))
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_177, (58, ), (1, ))
    assert_size_stride(primals_178, (58, ), (1, ))
    assert_size_stride(primals_180, (58, ), (1, ))
    assert_size_stride(primals_181, (58, ), (1, ))
    assert_size_stride(primals_183, (58, ), (1, ))
    assert_size_stride(primals_184, (58, ), (1, ))
    assert_size_stride(primals_186, (58, ), (1, ))
    assert_size_stride(primals_187, (58, ), (1, ))
    assert_size_stride(primals_189, (58, ), (1, ))
    assert_size_stride(primals_190, (58, ), (1, ))
    assert_size_stride(primals_192, (58, ), (1, ))
    assert_size_stride(primals_193, (58, ), (1, ))
    assert_size_stride(primals_195, (58, ), (1, ))
    assert_size_stride(primals_196, (58, ), (1, ))
    assert_size_stride(primals_198, (58, ), (1, ))
    assert_size_stride(primals_199, (58, ), (1, ))
    assert_size_stride(primals_201, (58, ), (1, ))
    assert_size_stride(primals_202, (58, ), (1, ))
    assert_size_stride(primals_204, (58, ), (1, ))
    assert_size_stride(primals_205, (58, ), (1, ))
    assert_size_stride(primals_207, (58, ), (1, ))
    assert_size_stride(primals_208, (58, ), (1, ))
    assert_size_stride(primals_210, (58, ), (1, ))
    assert_size_stride(primals_211, (58, ), (1, ))
    assert_size_stride(primals_213, (58, ), (1, ))
    assert_size_stride(primals_214, (58, ), (1, ))
    assert_size_stride(primals_216, (116, ), (1, ))
    assert_size_stride(primals_217, (116, ), (1, ))
    assert_size_stride(primals_219, (116, ), (1, ))
    assert_size_stride(primals_220, (116, ), (1, ))
    assert_size_stride(primals_222, (116, ), (1, ))
    assert_size_stride(primals_223, (116, ), (1, ))
    assert_size_stride(primals_225, (116, ), (1, ))
    assert_size_stride(primals_226, (116, ), (1, ))
    assert_size_stride(primals_228, (116, ), (1, ))
    assert_size_stride(primals_229, (116, ), (1, ))
    assert_size_stride(primals_231, (116, ), (1, ))
    assert_size_stride(primals_232, (116, ), (1, ))
    assert_size_stride(primals_234, (116, ), (1, ))
    assert_size_stride(primals_235, (116, ), (1, ))
    assert_size_stride(primals_237, (116, ), (1, ))
    assert_size_stride(primals_238, (116, ), (1, ))
    assert_size_stride(primals_240, (116, ), (1, ))
    assert_size_stride(primals_241, (116, ), (1, ))
    assert_size_stride(primals_243, (116, ), (1, ))
    assert_size_stride(primals_244, (116, ), (1, ))
    assert_size_stride(primals_246, (116, ), (1, ))
    assert_size_stride(primals_247, (116, ), (1, ))
    assert_size_stride(primals_249, (116, ), (1, ))
    assert_size_stride(primals_250, (116, ), (1, ))
    assert_size_stride(primals_252, (116, ), (1, ))
    assert_size_stride(primals_253, (116, ), (1, ))
    assert_size_stride(primals_255, (116, ), (1, ))
    assert_size_stride(primals_256, (116, ), (1, ))
    assert_size_stride(primals_258, (116, ), (1, ))
    assert_size_stride(primals_259, (116, ), (1, ))
    assert_size_stride(primals_261, (116, ), (1, ))
    assert_size_stride(primals_262, (116, ), (1, ))
    assert_size_stride(primals_264, (116, ), (1, ))
    assert_size_stride(primals_265, (116, ), (1, ))
    assert_size_stride(primals_267, (116, ), (1, ))
    assert_size_stride(primals_268, (116, ), (1, ))
    assert_size_stride(primals_270, (116, ), (1, ))
    assert_size_stride(primals_271, (116, ), (1, ))
    assert_size_stride(primals_273, (116, ), (1, ))
    assert_size_stride(primals_274, (116, ), (1, ))
    assert_size_stride(primals_276, (116, ), (1, ))
    assert_size_stride(primals_277, (116, ), (1, ))
    assert_size_stride(primals_279, (116, ), (1, ))
    assert_size_stride(primals_280, (116, ), (1, ))
    assert_size_stride(primals_282, (116, ), (1, ))
    assert_size_stride(primals_283, (116, ), (1, ))
    assert_size_stride(primals_285, (116, ), (1, ))
    assert_size_stride(primals_286, (116, ), (1, ))
    assert_size_stride(primals_288, (116, ), (1, ))
    assert_size_stride(primals_289, (116, ), (1, ))
    assert_size_stride(primals_291, (116, ), (1, ))
    assert_size_stride(primals_292, (116, ), (1, ))
    assert_size_stride(primals_294, (232, ), (1, ))
    assert_size_stride(primals_295, (232, ), (1, ))
    assert_size_stride(primals_297, (232, ), (1, ))
    assert_size_stride(primals_298, (232, ), (1, ))
    assert_size_stride(primals_300, (232, ), (1, ))
    assert_size_stride(primals_301, (232, ), (1, ))
    assert_size_stride(primals_303, (232, ), (1, ))
    assert_size_stride(primals_304, (232, ), (1, ))
    assert_size_stride(primals_306, (232, ), (1, ))
    assert_size_stride(primals_307, (232, ), (1, ))
    assert_size_stride(primals_309, (232, ), (1, ))
    assert_size_stride(primals_310, (232, ), (1, ))
    assert_size_stride(primals_312, (232, ), (1, ))
    assert_size_stride(primals_313, (232, ), (1, ))
    assert_size_stride(primals_315, (232, ), (1, ))
    assert_size_stride(primals_316, (232, ), (1, ))
    assert_size_stride(primals_318, (232, ), (1, ))
    assert_size_stride(primals_319, (232, ), (1, ))
    assert_size_stride(primals_321, (232, ), (1, ))
    assert_size_stride(primals_322, (232, ), (1, ))
    assert_size_stride(primals_324, (232, ), (1, ))
    assert_size_stride(primals_325, (232, ), (1, ))
    assert_size_stride(primals_327, (232, ), (1, ))
    assert_size_stride(primals_328, (232, ), (1, ))
    assert_size_stride(primals_330, (232, ), (1, ))
    assert_size_stride(primals_331, (232, ), (1, ))
    assert_size_stride(primals_333, (232, ), (1, ))
    assert_size_stride(primals_334, (232, ), (1, ))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_339, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(relu, (4, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(getitem, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(getitem_1, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_1, (4, 24, 28, 28), (18816, 1, 672, 24))
    assert_size_stride(add_3, (4, 24, 28, 28), (18816, 1, 672, 24))
    assert_size_stride(convolution_2, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_3, (4, 58, 56, 56), (181888, 1, 3248, 58))
    assert_size_stride(relu_2, (4, 58, 56, 56), (181888, 1, 3248, 58))
    assert_size_stride(convolution_4, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_9, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_5, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(getitem_3, (4, 58, 28, 28), (90944, 784, 28, 1))
    assert_size_stride(convolution_6, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(relu_4, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_7, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_15, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_8, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(getitem_5, (4, 58, 28, 28), (90944, 784, 28, 1))
    assert_size_stride(convolution_9, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(relu_6, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_10, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_21, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_11, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(getitem_7, (4, 58, 28, 28), (90944, 784, 28, 1))
    assert_size_stride(convolution_12, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(relu_8, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_13, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_27, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_14, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(view_7, (4, 116, 28, 28), (90944, 1, 3248, 116))
    assert_size_stride(convolution_15, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_31, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_16, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_17, (4, 116, 28, 28), (90944, 1, 3248, 116))
    assert_size_stride(relu_11, (4, 116, 28, 28), (90944, 1, 3248, 116))
    assert_size_stride(convolution_18, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_37, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_19, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_9, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_20, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_13, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_21, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_43, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_22, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_11, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_23, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_15, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_24, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_49, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_25, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_13, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_26, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_17, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_27, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_55, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_28, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_15, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_29, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_19, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_30, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_61, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_31, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_17, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_32, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_21, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_33, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_67, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_34, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_19, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_35, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_23, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_36, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_73, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_37, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_21, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_38, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_25, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_39, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_79, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_40, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(view_23, (4, 232, 14, 14), (45472, 1, 3248, 232))
    assert_size_stride(convolution_41, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_83, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_42, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_43, (4, 232, 14, 14), (45472, 1, 3248, 232))
    assert_size_stride(relu_28, (4, 232, 14, 14), (45472, 1, 3248, 232))
    assert_size_stride(convolution_44, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_89, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_45, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(getitem_23, (4, 232, 7, 7), (22736, 49, 7, 1))
    assert_size_stride(convolution_46, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(relu_30, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_47, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_95, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_48, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(getitem_25, (4, 232, 7, 7), (22736, 49, 7, 1))
    assert_size_stride(convolution_49, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(relu_32, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_50, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_101, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_51, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(getitem_27, (4, 232, 7, 7), (22736, 49, 7, 1))
    assert_size_stride(convolution_52, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(relu_34, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_53, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_107, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_54, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(view_31, (4, 464, 7, 7), (22736, 1, 3248, 464))
    assert_size_stride(convolution_55, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(mean, (4, 1024), (1024, 1))
    assert_size_stride(permute_17, (1000, 1024), (1024, 1))
    assert_size_stride(le, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(le_1, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_3, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_5, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_7, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_9, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_10, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_12, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_14, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_16, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_18, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_20, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_22, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_24, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_26, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_27, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_29, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_31, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_33, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_35, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_17, out=buf0)
    del permute_17
    buf1 = empty((1000, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), mean, out=buf1)
    del mean
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0(c_void_p(buf5.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del convolution_55
    del le
    del primals_167
    del primals_336
    del primals_337
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, view_31, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del primals_166
    del view_31
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((232, ), device='cpu', dtype=torch.float32)
    buf11 = empty((232, ), device='cpu', dtype=torch.float32)
    buf12 = buf11; del buf11  # reuse
    buf13 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(buf12.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf13.data_ptr()))
    del convolution_54
    del le_1
    del primals_164
    del primals_333
    del primals_334
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf14 = aten.convolution_backward(buf13, add_107, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_107
    del buf13
    del primals_163
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty((232, ), device='cpu', dtype=torch.float32)
    buf18 = empty((232, ), device='cpu', dtype=torch.float32)
    buf19 = buf18; del buf18  # reuse
    buf20 = buf15; del buf15  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_2(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf17.data_ptr()))
    del convolution_53
    del primals_161
    del primals_330
    del primals_331
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf21 = aten.convolution_backward(buf20, relu_34, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
    del buf20
    del primals_160
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = empty((232, ), device='cpu', dtype=torch.float32)
    buf25 = empty((232, ), device='cpu', dtype=torch.float32)
    buf26 = buf25; del buf25  # reuse
    buf27 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf24.data_ptr()))
    del convolution_52
    del primals_158
    del primals_327
    del primals_328
    del relu_34
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf28 = aten.convolution_backward(buf27, getitem_27, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_27
    del primals_157
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = empty((232, ), device='cpu', dtype=torch.float32)
    buf32 = empty((232, ), device='cpu', dtype=torch.float32)
    buf33 = buf32; del buf32  # reuse
    buf34 = buf27; del buf27  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf33.data_ptr()), c_void_p(le_3.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf34.data_ptr()))
    del convolution_51
    del le_3
    del primals_155
    del primals_324
    del primals_325
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf35 = aten.convolution_backward(buf34, add_101, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_101
    del buf34
    del primals_154
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = empty((232, ), device='cpu', dtype=torch.float32)
    buf39 = empty((232, ), device='cpu', dtype=torch.float32)
    buf40 = buf39; del buf39  # reuse
    buf41 = buf36; del buf36  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_5(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_50
    del primals_152
    del primals_321
    del primals_322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf42 = aten.convolution_backward(buf41, relu_32, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
    del primals_151
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = empty((232, ), device='cpu', dtype=torch.float32)
    buf46 = empty((232, ), device='cpu', dtype=torch.float32)
    buf47 = buf46; del buf46  # reuse
    buf48 = buf43; del buf43  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf45.data_ptr()))
    del convolution_49
    del primals_149
    del primals_318
    del primals_319
    del relu_32
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf49 = aten.convolution_backward(buf48, getitem_25, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_25
    del primals_148
    buf50 = buf49[0]
    buf51 = buf49[1]
    del buf49
    buf52 = reinterpret_tensor(buf48, (4, 232, 7, 7), (11368, 49, 7, 1), 0); del buf48  # reuse
    buf56 = buf41; del buf41  # reuse
    buf53 = empty((232, ), device='cpu', dtype=torch.float32)
    buf54 = empty((232, ), device='cpu', dtype=torch.float32)
    buf55 = buf54; del buf54  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf55.data_ptr()), c_void_p(le_5.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf53.data_ptr()))
    del buf52
    del convolution_48
    del le_5
    del primals_146
    del primals_315
    del primals_316
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf57 = aten.convolution_backward(buf56, add_95, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_95
    del buf56
    del primals_145
    buf58 = buf57[0]
    buf59 = buf57[1]
    del buf57
    buf60 = empty((232, ), device='cpu', dtype=torch.float32)
    buf61 = empty((232, ), device='cpu', dtype=torch.float32)
    buf62 = buf61; del buf61  # reuse
    buf63 = buf58; del buf58  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_8(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf60.data_ptr()))
    del convolution_47
    del primals_143
    del primals_312
    del primals_313
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf64 = aten.convolution_backward(buf63, relu_30, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
    del buf63
    del primals_142
    buf65 = buf64[0]
    buf66 = buf64[1]
    del buf64
    buf67 = empty((232, ), device='cpu', dtype=torch.float32)
    buf68 = empty((232, ), device='cpu', dtype=torch.float32)
    buf69 = buf68; del buf68  # reuse
    buf70 = buf65; del buf65  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf67.data_ptr()))
    del convolution_46
    del primals_140
    del primals_309
    del primals_310
    del relu_30
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf71 = aten.convolution_backward(buf70, getitem_23, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_23
    del primals_139
    buf72 = buf71[0]
    buf73 = buf71[1]
    del buf71
    buf74 = empty((4, 464, 7, 7), device='cpu', dtype=torch.float32)
    buf75 = empty((232, ), device='cpu', dtype=torch.float32)
    buf76 = empty((232, ), device='cpu', dtype=torch.float32)
    buf77 = buf76; del buf76  # reuse
    buf78 = buf70; del buf70  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf77.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(le_7.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf78.data_ptr()))
    del buf29
    del buf50
    del buf72
    del buf8
    del convolution_45
    del le_7
    del primals_137
    del primals_306
    del primals_307
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf79 = aten.convolution_backward(buf78, add_89, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_89
    del buf78
    del primals_136
    buf80 = buf79[0]
    buf81 = buf79[1]
    del buf79
    buf82 = empty((232, ), device='cpu', dtype=torch.float32)
    buf83 = empty((232, ), device='cpu', dtype=torch.float32)
    buf84 = buf83; del buf83  # reuse
    buf85 = buf80; del buf80  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_11(c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf82.data_ptr()))
    del convolution_44
    del primals_134
    del primals_303
    del primals_304
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf86 = aten.convolution_backward(buf85, relu_28, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
    del primals_133
    buf87 = buf86[0]
    buf88 = buf86[1]
    del buf86
    buf89 = empty((232, ), device='cpu', dtype=torch.float32)
    buf90 = empty((232, ), device='cpu', dtype=torch.float32)
    buf91 = buf90; del buf90  # reuse
    buf92 = buf87; del buf87  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf89.data_ptr()))
    del convolution_43
    del primals_131
    del primals_300
    del primals_301
    del relu_28
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf93 = aten.convolution_backward(buf92, view_23, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_130
    buf94 = buf93[0]
    buf95 = buf93[1]
    del buf93
    buf96 = empty((232, ), device='cpu', dtype=torch.float32)
    buf97 = empty((232, ), device='cpu', dtype=torch.float32)
    buf98 = buf97; del buf97  # reuse
    buf99 = buf85; del buf85  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf98.data_ptr()), c_void_p(le_9.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf99.data_ptr()))
    del convolution_42
    del le_9
    del primals_128
    del primals_297
    del primals_298
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf100 = aten.convolution_backward(buf99, add_83, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_83
    del buf99
    del primals_127
    buf101 = buf100[0]
    buf102 = buf100[1]
    del buf100
    buf103 = empty((232, ), device='cpu', dtype=torch.float32)
    buf104 = empty((232, ), device='cpu', dtype=torch.float32)
    buf105 = buf104; del buf104  # reuse
    buf106 = buf101; del buf101  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_14(c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf103.data_ptr()))
    del convolution_41
    del primals_125
    del primals_294
    del primals_295
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf107 = aten.convolution_backward(buf106, view_23, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
    del buf106
    del primals_124
    del view_23
    buf108 = buf107[0]
    buf109 = buf107[1]
    del buf107
    buf110 = empty((116, ), device='cpu', dtype=torch.float32)
    buf111 = empty((116, ), device='cpu', dtype=torch.float32)
    buf112 = buf111; del buf111  # reuse
    buf113 = reinterpret_tensor(buf74, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf74  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf112.data_ptr()), c_void_p(le_10.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf113.data_ptr()))
    del convolution_40
    del le_10
    del primals_122
    del primals_291
    del primals_292
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf114 = aten.convolution_backward(buf113, add_79, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_79
    del buf113
    del primals_121
    buf115 = buf114[0]
    buf116 = buf114[1]
    del buf114
    buf117 = empty((116, ), device='cpu', dtype=torch.float32)
    buf118 = empty((116, ), device='cpu', dtype=torch.float32)
    buf119 = buf118; del buf118  # reuse
    buf120 = buf115; del buf115  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_16(c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf117.data_ptr()))
    del convolution_39
    del primals_119
    del primals_288
    del primals_289
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf121 = aten.convolution_backward(buf120, relu_25, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del buf120
    del primals_118
    buf122 = buf121[0]
    buf123 = buf121[1]
    del buf121
    buf124 = empty((116, ), device='cpu', dtype=torch.float32)
    buf125 = empty((116, ), device='cpu', dtype=torch.float32)
    buf126 = buf125; del buf125  # reuse
    buf127 = buf122; del buf122  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf124.data_ptr()))
    del convolution_38
    del primals_116
    del primals_285
    del primals_286
    del relu_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf128 = aten.convolution_backward(buf127, getitem_21, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_21
    del primals_115
    buf129 = buf128[0]
    buf130 = buf128[1]
    del buf128
    buf131 = empty((116, ), device='cpu', dtype=torch.float32)
    buf132 = empty((116, ), device='cpu', dtype=torch.float32)
    buf133 = buf132; del buf132  # reuse
    buf134 = buf127; del buf127  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf133.data_ptr()), c_void_p(le_12.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf134.data_ptr()))
    del convolution_37
    del le_12
    del primals_113
    del primals_282
    del primals_283
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf135 = aten.convolution_backward(buf134, add_73, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_73
    del buf134
    del primals_112
    buf136 = buf135[0]
    buf137 = buf135[1]
    del buf135
    buf138 = empty((116, ), device='cpu', dtype=torch.float32)
    buf139 = empty((116, ), device='cpu', dtype=torch.float32)
    buf140 = buf139; del buf139  # reuse
    buf141 = buf136; del buf136  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_19(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf138.data_ptr()))
    del convolution_36
    del primals_110
    del primals_279
    del primals_280
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf142 = aten.convolution_backward(buf141, relu_23, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del buf141
    del primals_109
    buf143 = buf142[0]
    buf144 = buf142[1]
    del buf142
    buf145 = empty((116, ), device='cpu', dtype=torch.float32)
    buf146 = empty((116, ), device='cpu', dtype=torch.float32)
    buf147 = buf146; del buf146  # reuse
    buf148 = buf143; del buf143  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf145.data_ptr()))
    del convolution_35
    del primals_107
    del primals_276
    del primals_277
    del relu_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf149 = aten.convolution_backward(buf148, getitem_19, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_19
    del primals_106
    buf150 = buf149[0]
    buf151 = buf149[1]
    del buf149
    buf152 = buf148; del buf148  # reuse
    buf153 = empty((116, ), device='cpu', dtype=torch.float32)
    buf154 = empty((116, ), device='cpu', dtype=torch.float32)
    buf155 = buf154; del buf154  # reuse
    buf156 = buf152; del buf152  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(le_14.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf153.data_ptr()))
    del convolution_34
    del le_14
    del primals_104
    del primals_273
    del primals_274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf157 = aten.convolution_backward(buf156, add_67, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_67
    del buf156
    del primals_103
    buf158 = buf157[0]
    buf159 = buf157[1]
    del buf157
    buf160 = empty((116, ), device='cpu', dtype=torch.float32)
    buf161 = empty((116, ), device='cpu', dtype=torch.float32)
    buf162 = buf161; del buf161  # reuse
    buf163 = buf158; del buf158  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_22(c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf160.data_ptr()))
    del convolution_33
    del primals_101
    del primals_270
    del primals_271
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf164 = aten.convolution_backward(buf163, relu_21, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del buf163
    del primals_100
    buf165 = buf164[0]
    buf166 = buf164[1]
    del buf164
    buf167 = empty((116, ), device='cpu', dtype=torch.float32)
    buf168 = empty((116, ), device='cpu', dtype=torch.float32)
    buf169 = buf168; del buf168  # reuse
    buf170 = buf165; del buf165  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf167.data_ptr()))
    del convolution_32
    del primals_267
    del primals_268
    del primals_98
    del relu_21
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf171 = aten.convolution_backward(buf170, getitem_17, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_17
    del primals_97
    buf172 = buf171[0]
    buf173 = buf171[1]
    del buf171
    buf174 = reinterpret_tensor(buf92, (4, 232, 14, 14), (45472, 196, 14, 1), 0); del buf92  # reuse
    buf175 = empty((116, ), device='cpu', dtype=torch.float32)
    buf176 = empty((116, ), device='cpu', dtype=torch.float32)
    buf177 = buf176; del buf176  # reuse
    buf178 = buf170; del buf170  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf177.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(le_16.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf178.data_ptr()))
    del buf108
    del buf129
    del buf150
    del buf172
    del convolution_31
    del le_16
    del primals_264
    del primals_265
    del primals_95
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf179 = aten.convolution_backward(buf178, add_61, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_61
    del buf178
    del primals_94
    buf180 = buf179[0]
    buf181 = buf179[1]
    del buf179
    buf182 = empty((116, ), device='cpu', dtype=torch.float32)
    buf183 = empty((116, ), device='cpu', dtype=torch.float32)
    buf184 = buf183; del buf183  # reuse
    buf185 = buf180; del buf180  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_25(c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf182.data_ptr()))
    del convolution_30
    del primals_261
    del primals_262
    del primals_92
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf186 = aten.convolution_backward(buf185, relu_19, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del buf185
    del primals_91
    buf187 = buf186[0]
    buf188 = buf186[1]
    del buf186
    buf189 = empty((116, ), device='cpu', dtype=torch.float32)
    buf190 = empty((116, ), device='cpu', dtype=torch.float32)
    buf191 = buf190; del buf190  # reuse
    buf192 = buf187; del buf187  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf189.data_ptr()))
    del convolution_29
    del primals_258
    del primals_259
    del primals_89
    del relu_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf193 = aten.convolution_backward(buf192, getitem_15, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_15
    del primals_88
    buf194 = buf193[0]
    buf195 = buf193[1]
    del buf193
    buf196 = empty((116, ), device='cpu', dtype=torch.float32)
    buf197 = empty((116, ), device='cpu', dtype=torch.float32)
    buf198 = buf197; del buf197  # reuse
    buf199 = buf192; del buf192  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf198.data_ptr()), c_void_p(le_18.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf199.data_ptr()))
    del convolution_28
    del le_18
    del primals_255
    del primals_256
    del primals_86
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf200 = aten.convolution_backward(buf199, add_55, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_55
    del buf199
    del primals_85
    buf201 = buf200[0]
    buf202 = buf200[1]
    del buf200
    buf203 = empty((116, ), device='cpu', dtype=torch.float32)
    buf204 = empty((116, ), device='cpu', dtype=torch.float32)
    buf205 = buf204; del buf204  # reuse
    buf206 = buf201; del buf201  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_28(c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf203.data_ptr()))
    del convolution_27
    del primals_252
    del primals_253
    del primals_83
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf207 = aten.convolution_backward(buf206, relu_17, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del primals_82
    buf208 = buf207[0]
    buf209 = buf207[1]
    del buf207
    buf210 = empty((116, ), device='cpu', dtype=torch.float32)
    buf211 = empty((116, ), device='cpu', dtype=torch.float32)
    buf212 = buf211; del buf211  # reuse
    buf213 = buf208; del buf208  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf210.data_ptr()))
    del convolution_26
    del primals_249
    del primals_250
    del primals_80
    del relu_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf214 = aten.convolution_backward(buf213, getitem_13, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_13
    del primals_79
    buf215 = buf214[0]
    buf216 = buf214[1]
    del buf214
    buf217 = reinterpret_tensor(buf213, (4, 116, 14, 14), (22736, 196, 14, 1), 0); del buf213  # reuse
    buf221 = buf206; del buf206  # reuse
    buf218 = empty((116, ), device='cpu', dtype=torch.float32)
    buf219 = empty((116, ), device='cpu', dtype=torch.float32)
    buf220 = buf219; del buf219  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf220.data_ptr()), c_void_p(le_20.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf218.data_ptr()))
    del buf217
    del convolution_25
    del le_20
    del primals_246
    del primals_247
    del primals_77
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf222 = aten.convolution_backward(buf221, add_49, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_49
    del buf221
    del primals_76
    buf223 = buf222[0]
    buf224 = buf222[1]
    del buf222
    buf225 = empty((116, ), device='cpu', dtype=torch.float32)
    buf226 = empty((116, ), device='cpu', dtype=torch.float32)
    buf227 = buf226; del buf226  # reuse
    buf228 = buf223; del buf223  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_31(c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf225.data_ptr()))
    del convolution_24
    del primals_243
    del primals_244
    del primals_74
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf229 = aten.convolution_backward(buf228, relu_15, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del buf228
    del primals_73
    buf230 = buf229[0]
    buf231 = buf229[1]
    del buf229
    buf232 = empty((116, ), device='cpu', dtype=torch.float32)
    buf233 = empty((116, ), device='cpu', dtype=torch.float32)
    buf234 = buf233; del buf233  # reuse
    buf235 = buf230; del buf230  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf232.data_ptr()))
    del convolution_23
    del primals_240
    del primals_241
    del primals_71
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf236 = aten.convolution_backward(buf235, getitem_11, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_11
    del primals_70
    buf237 = buf236[0]
    buf238 = buf236[1]
    del buf236
    buf239 = reinterpret_tensor(buf94, (4, 232, 14, 14), (45472, 196, 14, 1), 0); del buf94  # reuse
    buf240 = empty((116, ), device='cpu', dtype=torch.float32)
    buf241 = empty((116, ), device='cpu', dtype=torch.float32)
    buf242 = buf241; del buf241  # reuse
    buf243 = buf235; del buf235  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf242.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(le_22.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()))
    del buf174
    del buf194
    del buf215
    del buf237
    del convolution_22
    del le_22
    del primals_237
    del primals_238
    del primals_68
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf244 = aten.convolution_backward(buf243, add_43, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_43
    del buf243
    del primals_67
    buf245 = buf244[0]
    buf246 = buf244[1]
    del buf244
    buf247 = empty((116, ), device='cpu', dtype=torch.float32)
    buf248 = empty((116, ), device='cpu', dtype=torch.float32)
    buf249 = buf248; del buf248  # reuse
    buf250 = buf245; del buf245  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_34(c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf247.data_ptr()))
    del convolution_21
    del primals_234
    del primals_235
    del primals_65
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf251 = aten.convolution_backward(buf250, relu_13, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del buf250
    del primals_64
    buf252 = buf251[0]
    buf253 = buf251[1]
    del buf251
    buf254 = empty((116, ), device='cpu', dtype=torch.float32)
    buf255 = empty((116, ), device='cpu', dtype=torch.float32)
    buf256 = buf255; del buf255  # reuse
    buf257 = buf252; del buf252  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf254.data_ptr()))
    del convolution_20
    del primals_231
    del primals_232
    del primals_62
    del relu_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf258 = aten.convolution_backward(buf257, getitem_9, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_9
    del primals_61
    buf259 = buf258[0]
    buf260 = buf258[1]
    del buf258
    buf261 = empty((116, ), device='cpu', dtype=torch.float32)
    buf262 = empty((116, ), device='cpu', dtype=torch.float32)
    buf263 = buf262; del buf262  # reuse
    buf264 = buf257; del buf257  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf263.data_ptr()), c_void_p(le_24.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf264.data_ptr()))
    del convolution_19
    del le_24
    del primals_228
    del primals_229
    del primals_59
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf265 = aten.convolution_backward(buf264, add_37, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_37
    del buf264
    del primals_58
    buf266 = buf265[0]
    buf267 = buf265[1]
    del buf265
    buf268 = empty((116, ), device='cpu', dtype=torch.float32)
    buf269 = empty((116, ), device='cpu', dtype=torch.float32)
    buf270 = buf269; del buf269  # reuse
    buf271 = buf266; del buf266  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_37(c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf268.data_ptr()))
    del convolution_18
    del primals_225
    del primals_226
    del primals_56
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf272 = aten.convolution_backward(buf271, relu_11, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del primals_55
    buf273 = buf272[0]
    buf274 = buf272[1]
    del buf272
    buf275 = empty((116, ), device='cpu', dtype=torch.float32)
    buf276 = empty((116, ), device='cpu', dtype=torch.float32)
    buf277 = buf276; del buf276  # reuse
    buf278 = buf273; del buf273  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf275.data_ptr()))
    del convolution_17
    del primals_222
    del primals_223
    del primals_53
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf279 = aten.convolution_backward(buf278, view_7, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_52
    buf280 = buf279[0]
    buf281 = buf279[1]
    del buf279
    buf282 = empty((116, ), device='cpu', dtype=torch.float32)
    buf283 = empty((116, ), device='cpu', dtype=torch.float32)
    buf284 = buf283; del buf283  # reuse
    buf285 = buf271; del buf271  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf284.data_ptr()), c_void_p(le_26.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf285.data_ptr()))
    del buf259
    del convolution_16
    del le_26
    del primals_219
    del primals_220
    del primals_50
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf286 = aten.convolution_backward(buf285, add_31, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_31
    del buf285
    del primals_49
    buf287 = buf286[0]
    buf288 = buf286[1]
    del buf286
    buf289 = empty((116, ), device='cpu', dtype=torch.float32)
    buf290 = empty((116, ), device='cpu', dtype=torch.float32)
    buf291 = buf290; del buf290  # reuse
    buf292 = buf287; del buf287  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_40(c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf289.data_ptr()))
    del convolution_15
    del primals_216
    del primals_217
    del primals_47
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf293 = aten.convolution_backward(buf292, view_7, primals_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
    del buf292
    del primals_46
    del view_7
    buf294 = buf293[0]
    buf295 = buf293[1]
    del buf293
    buf296 = empty((58, ), device='cpu', dtype=torch.float32)
    buf297 = empty((58, ), device='cpu', dtype=torch.float32)
    buf298 = buf297; del buf297  # reuse
    buf299 = reinterpret_tensor(buf239, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf239  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf298.data_ptr()), c_void_p(le_27.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf299.data_ptr()))
    del convolution_14
    del le_27
    del primals_213
    del primals_214
    del primals_44
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf300 = aten.convolution_backward(buf299, add_27, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_27
    del buf299
    del primals_43
    buf301 = buf300[0]
    buf302 = buf300[1]
    del buf300
    buf303 = empty((58, ), device='cpu', dtype=torch.float32)
    buf304 = empty((58, ), device='cpu', dtype=torch.float32)
    buf305 = buf304; del buf304  # reuse
    buf306 = buf301; del buf301  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_42(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf303.data_ptr()))
    del convolution_13
    del primals_210
    del primals_211
    del primals_41
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf307 = aten.convolution_backward(buf306, relu_8, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
    del buf306
    del primals_40
    buf308 = buf307[0]
    buf309 = buf307[1]
    del buf307
    buf310 = empty((58, ), device='cpu', dtype=torch.float32)
    buf311 = empty((58, ), device='cpu', dtype=torch.float32)
    buf312 = buf311; del buf311  # reuse
    buf313 = buf308; del buf308  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf310.data_ptr()))
    del convolution_12
    del primals_207
    del primals_208
    del primals_38
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf314 = aten.convolution_backward(buf313, getitem_7, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_7
    del primals_37
    buf315 = buf314[0]
    buf316 = buf314[1]
    del buf314
    buf317 = empty((58, ), device='cpu', dtype=torch.float32)
    buf318 = empty((58, ), device='cpu', dtype=torch.float32)
    buf319 = buf318; del buf318  # reuse
    buf320 = buf313; del buf313  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44(c_void_p(buf319.data_ptr()), c_void_p(le_29.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf320.data_ptr()))
    del convolution_11
    del le_29
    del primals_204
    del primals_205
    del primals_35
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf321 = aten.convolution_backward(buf320, add_21, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_21
    del buf320
    del primals_34
    buf322 = buf321[0]
    buf323 = buf321[1]
    del buf321
    buf324 = empty((58, ), device='cpu', dtype=torch.float32)
    buf325 = empty((58, ), device='cpu', dtype=torch.float32)
    buf326 = buf325; del buf325  # reuse
    buf327 = buf322; del buf322  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_45(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf324.data_ptr()))
    del convolution_10
    del primals_201
    del primals_202
    del primals_32
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf328 = aten.convolution_backward(buf327, relu_6, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
    del buf327
    del primals_31
    buf329 = buf328[0]
    buf330 = buf328[1]
    del buf328
    buf331 = empty((58, ), device='cpu', dtype=torch.float32)
    buf332 = empty((58, ), device='cpu', dtype=torch.float32)
    buf333 = buf332; del buf332  # reuse
    buf334 = buf329; del buf329  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_46(c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf331.data_ptr()))
    del convolution_9
    del primals_198
    del primals_199
    del primals_29
    del relu_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf335 = aten.convolution_backward(buf334, getitem_5, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_5
    del primals_28
    buf336 = buf335[0]
    buf337 = buf335[1]
    del buf335
    buf338 = buf334; del buf334  # reuse
    buf339 = empty((58, ), device='cpu', dtype=torch.float32)
    buf340 = empty((58, ), device='cpu', dtype=torch.float32)
    buf341 = buf340; del buf340  # reuse
    buf342 = buf338; del buf338  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47(c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(le_31.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf339.data_ptr()))
    del convolution_8
    del le_31
    del primals_195
    del primals_196
    del primals_26
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf343 = aten.convolution_backward(buf342, add_15, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_15
    del buf342
    del primals_25
    buf344 = buf343[0]
    buf345 = buf343[1]
    del buf343
    buf346 = empty((58, ), device='cpu', dtype=torch.float32)
    buf347 = empty((58, ), device='cpu', dtype=torch.float32)
    buf348 = buf347; del buf347  # reuse
    buf349 = buf344; del buf344  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_48(c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf346.data_ptr()))
    del convolution_7
    del primals_192
    del primals_193
    del primals_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf350 = aten.convolution_backward(buf349, relu_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
    del buf349
    del primals_22
    buf351 = buf350[0]
    buf352 = buf350[1]
    del buf350
    buf353 = empty((58, ), device='cpu', dtype=torch.float32)
    buf354 = empty((58, ), device='cpu', dtype=torch.float32)
    buf355 = buf354; del buf354  # reuse
    buf356 = buf351; del buf351  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf353.data_ptr()))
    del convolution_6
    del primals_189
    del primals_190
    del primals_20
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf357 = aten.convolution_backward(buf356, getitem_3, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_3
    del primals_19
    buf358 = buf357[0]
    buf359 = buf357[1]
    del buf357
    buf360 = reinterpret_tensor(buf278, (4, 116, 28, 28), (90944, 784, 28, 1), 0); del buf278  # reuse
    buf361 = empty((58, ), device='cpu', dtype=torch.float32)
    buf362 = empty((58, ), device='cpu', dtype=torch.float32)
    buf363 = buf362; del buf362  # reuse
    buf364 = buf356; del buf356  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(buf363.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(le_33.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf364.data_ptr()))
    del buf280
    del buf294
    del buf315
    del buf336
    del buf358
    del convolution_5
    del le_33
    del primals_17
    del primals_186
    del primals_187
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf365 = aten.convolution_backward(buf364, add_9, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_9
    del buf364
    del primals_16
    buf366 = buf365[0]
    buf367 = buf365[1]
    del buf365
    buf368 = empty((58, ), device='cpu', dtype=torch.float32)
    buf369 = empty((58, ), device='cpu', dtype=torch.float32)
    buf370 = buf369; del buf369  # reuse
    buf371 = buf366; del buf366  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_51(c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf368.data_ptr()))
    del convolution_4
    del primals_14
    del primals_183
    del primals_184
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf372 = aten.convolution_backward(buf371, relu_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
    del primals_13
    buf373 = buf372[0]
    buf374 = buf372[1]
    del buf372
    buf375 = empty((58, ), device='cpu', dtype=torch.float32)
    buf376 = empty((58, ), device='cpu', dtype=torch.float32)
    buf377 = buf376; del buf376  # reuse
    buf378 = buf373; del buf373  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf375.data_ptr()))
    del convolution_3
    del primals_11
    del primals_180
    del primals_181
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf379 = aten.convolution_backward(buf378, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf378
    del primals_10
    buf380 = buf379[0]
    buf381 = buf379[1]
    del buf379
    buf382 = empty((58, ), device='cpu', dtype=torch.float32)
    buf383 = empty((58, ), device='cpu', dtype=torch.float32)
    buf384 = buf383; del buf383  # reuse
    buf385 = buf371; del buf371  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53(c_void_p(buf384.data_ptr()), c_void_p(le_35.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf385.data_ptr()))
    del buf360
    del convolution_2
    del le_35
    del primals_177
    del primals_178
    del primals_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf386 = aten.convolution_backward(buf385, add_3, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_3
    del buf385
    del primals_7
    buf387 = buf386[0]
    buf388 = buf386[1]
    del buf386
    buf389 = empty((24, ), device='cpu', dtype=torch.float32)
    buf390 = empty((24, ), device='cpu', dtype=torch.float32)
    buf391 = buf390; del buf390  # reuse
    buf392 = buf387; del buf387  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_54(c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf389.data_ptr()))
    del convolution_1
    del primals_174
    del primals_175
    del primals_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf393 = aten.convolution_backward(buf392, getitem, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False])
    del buf392
    del getitem
    del primals_4
    buf394 = buf393[0]
    buf395 = buf393[1]
    del buf393
    buf396 = buf380; del buf380  # reuse
    cpp_fused_add_55(c_void_p(buf396.data_ptr()), c_void_p(buf394.data_ptr()))
    del buf394
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf397 = aten.max_pool2d_with_indices_backward(buf396, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1)
    del buf396
    del getitem_1
    buf398 = buf397
    del buf397
    buf399 = empty((24, ), device='cpu', dtype=torch.float32)
    buf400 = empty((24, ), device='cpu', dtype=torch.float32)
    buf401 = buf400; del buf400  # reuse
    buf402 = buf398; del buf398  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56(c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf399.data_ptr()))
    del convolution
    del primals_171
    del primals_172
    del primals_2
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf403 = aten.convolution_backward(buf402, primals_339, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf402
    del primals_1
    del primals_339
    buf404 = buf403[1]
    return (buf404, buf401, buf399, buf395, buf391, buf389, buf388, buf384, buf382, buf381, buf377, buf375, buf374, buf370, buf368, buf367, buf363, buf361, buf359, buf355, buf353, buf352, buf348, buf346, buf345, buf341, buf339, buf337, buf333, buf331, buf330, buf326, buf324, buf323, buf319, buf317, buf316, buf312, buf310, buf309, buf305, buf303, buf302, buf298, buf296, buf295, buf291, buf289, buf288, buf284, buf282, buf281, buf277, buf275, buf274, buf270, buf268, buf267, buf263, buf261, buf260, buf256, buf254, buf253, buf249, buf247, buf246, buf242, buf240, buf238, buf234, buf232, buf231, buf227, buf225, buf224, buf220, buf218, buf216, buf212, buf210, buf209, buf205, buf203, buf202, buf198, buf196, buf195, buf191, buf189, buf188, buf184, buf182, buf181, buf177, buf175, buf173, buf169, buf167, buf166, buf162, buf160, buf159, buf155, buf153, buf151, buf147, buf145, buf144, buf140, buf138, buf137, buf133, buf131, buf130, buf126, buf124, buf123, buf119, buf117, buf116, buf112, buf110, buf109, buf105, buf103, buf102, buf98, buf96, buf95, buf91, buf89, buf88, buf84, buf82, buf81, buf77, buf75, buf73, buf69, buf67, buf66, buf62, buf60, buf59, buf55, buf53, buf51, buf47, buf45, buf44, buf40, buf38, buf37, buf33, buf31, buf30, buf26, buf24, buf23, buf19, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1024, 464, 1, 1), (464, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((116, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((232, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    getitem = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.int64)
    convolution_1 = rand_strided((4, 24, 28, 28), (18816, 1, 672, 24), device='cpu', dtype=torch.float32)
    add_3 = rand_strided((4, 24, 28, 28), (18816, 1, 672, 24), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    add_9 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((4, 58, 28, 28), (90944, 784, 28, 1), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    add_15 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((4, 58, 28, 28), (90944, 784, 28, 1), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    add_21 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((4, 58, 28, 28), (90944, 784, 28, 1), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    add_27 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_31 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_37 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_43 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_49 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_55 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    getitem_15 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_61 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_67 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_73 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    add_79 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    add_83 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    add_89 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((4, 232, 7, 7), (22736, 49, 7, 1), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    add_95 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((4, 232, 7, 7), (22736, 49, 7, 1), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    add_101 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((4, 232, 7, 7), (22736, 49, 7, 1), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    add_107 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((4, 464, 7, 7), (22736, 1, 3248, 464), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    mean = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_17 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    le_1 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    le_3 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    le_5 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    le_7 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    le_9 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cpu', dtype=torch.bool)
    le_10 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_12 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_14 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_16 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_18 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_20 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_22 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_24 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_26 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cpu', dtype=torch.bool)
    le_27 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    le_29 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    le_31 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    le_33 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    le_35 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, primals_339, convolution, relu, getitem, getitem_1, convolution_1, add_3, convolution_2, convolution_3, relu_2, convolution_4, add_9, convolution_5, getitem_3, convolution_6, relu_4, convolution_7, add_15, convolution_8, getitem_5, convolution_9, relu_6, convolution_10, add_21, convolution_11, getitem_7, convolution_12, relu_8, convolution_13, add_27, convolution_14, view_7, convolution_15, add_31, convolution_16, convolution_17, relu_11, convolution_18, add_37, convolution_19, getitem_9, convolution_20, relu_13, convolution_21, add_43, convolution_22, getitem_11, convolution_23, relu_15, convolution_24, add_49, convolution_25, getitem_13, convolution_26, relu_17, convolution_27, add_55, convolution_28, getitem_15, convolution_29, relu_19, convolution_30, add_61, convolution_31, getitem_17, convolution_32, relu_21, convolution_33, add_67, convolution_34, getitem_19, convolution_35, relu_23, convolution_36, add_73, convolution_37, getitem_21, convolution_38, relu_25, convolution_39, add_79, convolution_40, view_23, convolution_41, add_83, convolution_42, convolution_43, relu_28, convolution_44, add_89, convolution_45, getitem_23, convolution_46, relu_30, convolution_47, add_95, convolution_48, getitem_25, convolution_49, relu_32, convolution_50, add_101, convolution_51, getitem_27, convolution_52, relu_34, convolution_53, add_107, convolution_54, view_31, convolution_55, mean, permute_17, le, le_1, le_3, le_5, le_7, le_9, le_10, le_12, le_14, le_16, le_18, le_20, le_22, le_24, le_26, le_27, le_29, le_31, le_33, le_35, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('shufflenet_v2_x1_0', benchmark_compiled_module)
