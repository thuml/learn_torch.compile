
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


cpp_fused_sigmoid_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp3 - tmp1;
            auto tmp5 = tmp1 * tmp4;
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(256L); x0<static_cast<long>(260L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr1[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1.0);
            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_sum_threshold_backward_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(65L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(130L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(195L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(64L); x0<static_cast<long>(65L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr0[static_cast<long>(65L + x0)];
            auto tmp3 = in_ptr0[static_cast<long>(130L + x0)];
            auto tmp5 = in_ptr0[static_cast<long>(195L + x0)];
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
            out_ptr0[static_cast<long>(x0)] = tmp6;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x0)));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = tmp4 / 16;
                    auto tmp6 = static_cast<int>(0);
                    auto tmp7 = static_cast<int>(1);
                    auto tmp8 = tmp6 < tmp7;
                    auto tmp9 = tmp8 & tmp8;
                    auto tmp10 = to_float_mask(tmp9);
                    auto tmp11 = decltype(tmp5)::blendv(tmp2, tmp5, tmp10);
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                    auto tmp14 = static_cast<float>(1e-05);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 + tmp15;
                    auto tmp17 = tmp16.rsqrt();
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp20 = tmp12 * tmp19;
                    tmp20.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                tmp13.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
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
    {
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
                float tmp_acc4 = 0;
                at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x2) + (8192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x2) + (8192L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x2) + (8192L*x1)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x2) + (8192L*x1)));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x2) + (8192L*x1)));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x2) + (8192L*x1)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = tmp4 / 16;
                        auto tmp6 = static_cast<int>(0);
                        auto tmp7 = static_cast<int>(1);
                        auto tmp8 = tmp6 < tmp7;
                        auto tmp9 = tmp8 & tmp8;
                        auto tmp10 = to_float_mask(tmp9);
                        auto tmp11 = decltype(tmp5)::blendv(tmp2, tmp5, tmp10);
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp16 = tmp12 * tmp15;
                        auto tmp18 = to_float_mask(tmp17 <= tmp2);
                        auto tmp20 = tmp12 + tmp19;
                        auto tmp21 = decltype(tmp2)::blendv(tmp20, tmp2, tmp18);
                        auto tmp24 = tmp22 - tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = tmp26 - tmp27;
                        auto tmp29 = tmp21 * tmp28;
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                        tmp_acc1_vec = tmp_acc1_vec + tmp16;
                        tmp_acc2_vec = tmp_acc2_vec + tmp21;
                        tmp_acc3_vec = tmp_acc3_vec + tmp25;
                        tmp_acc4_vec = tmp_acc4_vec + tmp29;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
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


cpp_fused_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
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
                       float* out_ptr1)
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x0)));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = to_float_mask(tmp4 <= tmp2);
                    auto tmp7 = tmp6 / 16;
                    auto tmp8 = static_cast<int>(0);
                    auto tmp9 = static_cast<int>(1);
                    auto tmp10 = tmp8 < tmp9;
                    auto tmp11 = tmp10 & tmp10;
                    auto tmp12 = to_float_mask(tmp11);
                    auto tmp13 = decltype(tmp7)::blendv(tmp2, tmp7, tmp12);
                    auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp5);
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = decltype(tmp2)::blendv(tmp16, tmp2, tmp3);
                    auto tmp19 = static_cast<float>(1e-05);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 + tmp20;
                    auto tmp22 = tmp21.rsqrt();
                    auto tmp24 = tmp22 * tmp23;
                    auto tmp25 = tmp17 * tmp24;
                    auto tmp27 = tmp26 + tmp20;
                    auto tmp28 = tmp27.rsqrt();
                    auto tmp30 = tmp28 * tmp29;
                    auto tmp31 = tmp17 * tmp30;
                    tmp25.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                    tmp31.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_6 = async_compile.cpp('''
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp6 = tmp4 + tmp5;
                auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp11.rsqrt();
                auto tmp14 = tmp12 * tmp13;
                auto tmp15 = tmp7 * tmp14;
                tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                auto tmp10 = tmp9 + tmp3;
                auto tmp11 = tmp10.rsqrt();
                auto tmp13 = tmp11 * tmp12;
                auto tmp14 = tmp0 * tmp13;
                tmp8.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                tmp14.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_11 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp9 + tmp3;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    tmp14.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp9 + tmp3;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    tmp14.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16384L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, relu, convolution_1, relu_1, convolution_2, convolution_3, relu_2, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, convolution_11, relu_9, convolution_12, convolution_13, relu_10, convolution_14, relu_11, convolution_15, relu_12, convolution_16, relu_13, convolution_17, convolution_18, relu_14, convolution_19, relu_15, convolution_20, relu_16, view, sigmoid, permute_1, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 9, 3, 3), (81, 1, 27, 9))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_28, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_40, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_43, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_46, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_52, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_58, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_61, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (64, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_129, (4, 9, 128, 128), (147456, 1, 1152, 9))
    assert_size_stride(convolution, (4, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(relu, (4, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_1, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_1, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_2, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_3, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_2, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_4, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_3, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_5, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(relu_4, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_6, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_5, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_7, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_8, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_6, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_9, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_7, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_10, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(relu_8, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_11, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_9, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_12, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_13, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_10, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_14, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_11, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_15, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(relu_12, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_16, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_13, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_17, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_18, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_14, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_19, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_15, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(convolution_20, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(relu_16, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(view, (4, 512), (512, 1))
    assert_size_stride(sigmoid, (4, 65), (65, 1))
    assert_size_stride(permute_1, (65, 512), (512, 1))
    assert_size_stride(tangents_1, (4, 65), (65, 1))
    buf0 = empty((4, 65), device='cpu', dtype=torch.float32)
    cpp_fused_sigmoid_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(sigmoid.data_ptr()), c_void_p(buf0.data_ptr()))
    del sigmoid
    del tangents_1
    buf1 = empty((4, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf0, permute_1, out=buf1)
    del permute_1
    buf2 = empty((65, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (65, 4), (1, 65), 0), view, out=buf2)
    del view
    buf3 = empty((65, ), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_sum_threshold_backward_view_1(c_void_p(buf0.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf0
    del primals_62
    # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf8 = aten.convolution_backward(buf7, relu_15, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_61
    buf9 = buf8[0]
    buf14 = buf7; del buf7  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(relu_15.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf14.data_ptr()))
    del primals_59
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf15 = aten.convolution_backward(buf14, relu_14, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_58
    buf16 = buf15[0]
    buf4 = empty((512, ), device='cpu', dtype=torch.float32)
    buf5 = empty((512, ), device='cpu', dtype=torch.float32)
    buf18 = empty((512, ), device='cpu', dtype=torch.float32)
    buf19 = empty((512, ), device='cpu', dtype=torch.float32)
    buf25 = empty((512, ), device='cpu', dtype=torch.float32)
    buf6 = buf5; del buf5  # reuse
    cpp_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf6.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf25.data_ptr()))
    del convolution_17
    del convolution_18
    del convolution_20
    del primals_117
    del primals_120
    del primals_126
    del primals_127
    buf10 = buf8[1]
    del buf8
    buf11 = empty((512, ), device='cpu', dtype=torch.float32)
    buf12 = empty((512, ), device='cpu', dtype=torch.float32)
    buf13 = buf12; del buf12  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_4(c_void_p(buf13.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf11.data_ptr()))
    del convolution_19
    del primals_123
    del primals_124
    del relu_15
    buf17 = buf15[1]
    del buf15
    buf20 = buf19; del buf19  # reuse
    buf21 = buf9; del buf9  # reuse
    buf27 = buf14; del buf14  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf20.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf27.data_ptr()))
    del buf1
    del buf16
    del primals_121
    del primals_53
    del primals_56
    del relu_14
    del relu_16
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf22 = aten.convolution_backward(buf21, relu_12, primals_55, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf21
    del primals_55
    buf23 = buf22[0]
    buf24 = buf22[1]
    del buf22
    buf26 = buf25; del buf25  # reuse
    cpp_fused_native_batch_norm_backward_6(c_void_p(buf26.data_ptr()), c_void_p(primals_118.data_ptr()))
    del primals_118
    # Source Nodes: [], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf28 = aten.convolution_backward(buf27, relu_13, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf27
    del primals_52
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = empty((512, ), device='cpu', dtype=torch.float32)
    buf32 = empty((512, ), device='cpu', dtype=torch.float32)
    buf33 = buf32; del buf32  # reuse
    buf34 = buf29; del buf29  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf31.data_ptr()))
    del convolution_16
    del primals_114
    del primals_115
    del primals_50
    del relu_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf35 = aten.convolution_backward(buf34, relu_12, primals_49, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf34
    del primals_49
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = empty((256, ), device='cpu', dtype=torch.float32)
    buf39 = empty((256, ), device='cpu', dtype=torch.float32)
    buf40 = buf39; del buf39  # reuse
    buf41 = empty_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(buf40.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf41.data_ptr()))
    del convolution_15
    del primals_111
    del primals_112
    del primals_47
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf42 = aten.convolution_backward(buf41, relu_11, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_46
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = empty((256, ), device='cpu', dtype=torch.float32)
    buf46 = empty((256, ), device='cpu', dtype=torch.float32)
    buf47 = buf46; del buf46  # reuse
    buf48 = buf43; del buf43  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf45.data_ptr()))
    del convolution_14
    del primals_108
    del primals_109
    del primals_44
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf49 = aten.convolution_backward(buf48, relu_10, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_43
    buf50 = buf49[0]
    buf51 = buf49[1]
    del buf49
    buf52 = buf23; del buf23  # reuse
    buf53 = empty((256, ), device='cpu', dtype=torch.float32)
    buf54 = empty((256, ), device='cpu', dtype=torch.float32)
    buf60 = empty((256, ), device='cpu', dtype=torch.float32)
    buf55 = buf54; del buf54  # reuse
    buf56 = buf48; del buf48  # reuse
    buf62 = buf41; del buf41  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf52.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf62.data_ptr()))
    del buf36
    del buf50
    del buf52
    del convolution_12
    del convolution_13
    del primals_102
    del primals_105
    del primals_106
    del primals_38
    del primals_41
    del relu_10
    del relu_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf57 = aten.convolution_backward(buf56, relu_8, primals_40, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf56
    del primals_40
    buf58 = buf57[0]
    buf59 = buf57[1]
    del buf57
    buf61 = buf60; del buf60  # reuse
    cpp_fused_native_batch_norm_backward_11(c_void_p(buf61.data_ptr()), c_void_p(primals_103.data_ptr()))
    del primals_103
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf63 = aten.convolution_backward(buf62, relu_9, primals_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf62
    del primals_37
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    buf66 = empty((256, ), device='cpu', dtype=torch.float32)
    buf67 = empty((256, ), device='cpu', dtype=torch.float32)
    buf68 = buf67; del buf67  # reuse
    buf69 = buf64; del buf64  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf66.data_ptr()))
    del convolution_11
    del primals_100
    del primals_35
    del primals_99
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf70 = aten.convolution_backward(buf69, relu_8, primals_34, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf69
    del primals_34
    buf71 = buf70[0]
    buf72 = buf70[1]
    del buf70
    buf73 = empty((128, ), device='cpu', dtype=torch.float32)
    buf74 = empty((128, ), device='cpu', dtype=torch.float32)
    buf75 = buf74; del buf74  # reuse
    buf76 = empty_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf75.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()))
    del convolution_10
    del primals_32
    del primals_96
    del primals_97
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf77 = aten.convolution_backward(buf76, relu_7, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_31
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf80 = empty((128, ), device='cpu', dtype=torch.float32)
    buf81 = empty((128, ), device='cpu', dtype=torch.float32)
    buf82 = buf81; del buf81  # reuse
    buf83 = buf78; del buf78  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf80.data_ptr()))
    del convolution_9
    del primals_29
    del primals_93
    del primals_94
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf84 = aten.convolution_backward(buf83, relu_6, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_28
    buf85 = buf84[0]
    buf86 = buf84[1]
    del buf84
    buf87 = buf58; del buf58  # reuse
    buf88 = empty((128, ), device='cpu', dtype=torch.float32)
    buf89 = empty((128, ), device='cpu', dtype=torch.float32)
    buf95 = empty((128, ), device='cpu', dtype=torch.float32)
    buf90 = buf89; del buf89  # reuse
    buf91 = buf83; del buf83  # reuse
    buf97 = buf76; del buf76  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf87.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf97.data_ptr()))
    del buf71
    del buf85
    del buf87
    del convolution_7
    del convolution_8
    del primals_23
    del primals_26
    del primals_87
    del primals_90
    del primals_91
    del relu_6
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf92 = aten.convolution_backward(buf91, relu_4, primals_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf91
    del primals_25
    buf93 = buf92[0]
    buf94 = buf92[1]
    del buf92
    buf96 = buf95; del buf95  # reuse
    cpp_fused_native_batch_norm_backward_16(c_void_p(buf96.data_ptr()), c_void_p(primals_88.data_ptr()))
    del primals_88
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf98 = aten.convolution_backward(buf97, relu_5, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf97
    del primals_22
    buf99 = buf98[0]
    buf100 = buf98[1]
    del buf98
    buf101 = empty((128, ), device='cpu', dtype=torch.float32)
    buf102 = empty((128, ), device='cpu', dtype=torch.float32)
    buf103 = buf102; del buf102  # reuse
    buf104 = buf99; del buf99  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf101.data_ptr()))
    del convolution_6
    del primals_20
    del primals_84
    del primals_85
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf105 = aten.convolution_backward(buf104, relu_4, primals_19, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf104
    del primals_19
    buf106 = buf105[0]
    buf107 = buf105[1]
    del buf105
    buf108 = empty((64, ), device='cpu', dtype=torch.float32)
    buf109 = empty((64, ), device='cpu', dtype=torch.float32)
    buf110 = buf109; del buf109  # reuse
    buf111 = empty_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf110.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf111.data_ptr()))
    del convolution_5
    del primals_17
    del primals_81
    del primals_82
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf112 = aten.convolution_backward(buf111, relu_3, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_16
    buf113 = buf112[0]
    buf114 = buf112[1]
    del buf112
    buf115 = empty((64, ), device='cpu', dtype=torch.float32)
    buf116 = empty((64, ), device='cpu', dtype=torch.float32)
    buf117 = buf116; del buf116  # reuse
    buf118 = buf113; del buf113  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf115.data_ptr()))
    del convolution_4
    del primals_14
    del primals_78
    del primals_79
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf119 = aten.convolution_backward(buf118, relu_2, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_13
    buf120 = buf119[0]
    buf121 = buf119[1]
    del buf119
    buf122 = buf106; del buf106  # reuse
    buf123 = empty((64, ), device='cpu', dtype=torch.float32)
    buf124 = empty((64, ), device='cpu', dtype=torch.float32)
    buf130 = empty((64, ), device='cpu', dtype=torch.float32)
    buf125 = buf124; del buf124  # reuse
    buf126 = buf118; del buf118  # reuse
    buf132 = buf111; del buf111  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf122.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf132.data_ptr()))
    del buf120
    del buf122
    del buf93
    del convolution_2
    del convolution_3
    del primals_11
    del primals_72
    del primals_75
    del primals_76
    del primals_8
    del relu_2
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf127 = aten.convolution_backward(buf126, relu, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf126
    del primals_10
    buf128 = buf127[0]
    buf129 = buf127[1]
    del buf127
    buf131 = buf130; del buf130  # reuse
    cpp_fused_native_batch_norm_backward_21(c_void_p(buf131.data_ptr()), c_void_p(primals_73.data_ptr()))
    del primals_73
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf133 = aten.convolution_backward(buf132, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf132
    del primals_7
    buf134 = buf133[0]
    buf135 = buf133[1]
    del buf133
    buf136 = empty((64, ), device='cpu', dtype=torch.float32)
    buf137 = empty((64, ), device='cpu', dtype=torch.float32)
    buf138 = buf137; del buf137  # reuse
    buf139 = buf134; del buf134  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf136.data_ptr()))
    del convolution_1
    del primals_5
    del primals_69
    del primals_70
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf140 = aten.convolution_backward(buf139, relu, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf139
    del primals_4
    buf141 = buf140[0]
    buf142 = buf140[1]
    del buf140
    buf143 = empty((64, ), device='cpu', dtype=torch.float32)
    buf144 = empty((64, ), device='cpu', dtype=torch.float32)
    buf145 = buf144; del buf144  # reuse
    buf146 = buf128; del buf128  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf143.data_ptr()))
    del buf141
    del convolution
    del primals_2
    del primals_66
    del primals_67
    del relu
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf147 = aten.convolution_backward(buf146, primals_129, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf146
    del primals_1
    del primals_129
    buf148 = buf147[1]
    return (buf148, buf145, buf143, buf142, buf138, buf136, buf135, buf131, buf123, buf129, buf125, buf123, buf121, buf117, buf115, buf114, buf110, buf108, buf107, buf103, buf101, buf100, buf96, buf88, buf94, buf90, buf88, buf86, buf82, buf80, buf79, buf75, buf73, buf72, buf68, buf66, buf65, buf61, buf53, buf59, buf55, buf53, buf51, buf47, buf45, buf44, buf40, buf38, buf37, buf33, buf31, buf30, buf26, buf18, buf24, buf20, buf18, buf17, buf13, buf11, buf10, buf6, buf4, reinterpret_tensor(buf2, (65, 512), (512, 1), 0), buf3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 9, 3, 3), (81, 1, 27, 9), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((4, 9, 128, 128), (147456, 1, 1152, 9), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    view = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.float32)
    sigmoid = rand_strided((4, 65), (65, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((65, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 65), (65, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, relu, convolution_1, relu_1, convolution_2, convolution_3, relu_2, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, convolution_11, relu_9, convolution_12, convolution_13, relu_10, convolution_14, relu_11, convolution_15, relu_12, convolution_16, relu_13, convolution_17, convolution_18, relu_14, convolution_19, relu_15, convolution_20, relu_16, view, sigmoid, permute_1, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LearningToPaint', benchmark_compiled_module)
