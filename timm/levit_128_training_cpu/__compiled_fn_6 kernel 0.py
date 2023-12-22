
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


cpp_fused__native_batch_norm_legit_functional_div_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(2.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            tmp3.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
}
''')


cpp_fused_native_batch_norm_backward_2 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp1;
                    tmp_acc2_vec = tmp_acc2_vec + tmp5;
                    tmp_acc3_vec = tmp_acc3_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            auto tmp11 = tmp10 * tmp8;
            tmp9.store(out_ptr4 + static_cast<long>(x0));
            tmp11.store(out_ptr5 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_3 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp26 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.125);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = static_cast<float>(8.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 / tmp10;
                auto tmp12 = static_cast<float>(1e-05);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 + tmp13;
                auto tmp15 = tmp14.rsqrt();
                auto tmp16 = tmp15 * tmp15;
                auto tmp17 = tmp7 * tmp16;
                auto tmp18 = tmp3 * tmp17;
                auto tmp19 = tmp0 - tmp18;
                auto tmp21 = tmp20 * tmp6;
                auto tmp22 = tmp19 - tmp21;
                auto tmp24 = tmp15 * tmp23;
                auto tmp25 = tmp22 * tmp24;
                auto tmp28 = tmp27 * tmp6;
                auto tmp29 = tmp28 * tmp16;
                auto tmp30 = tmp3 * tmp29;
                auto tmp31 = tmp26 - tmp30;
                auto tmp33 = tmp32 * tmp6;
                auto tmp34 = tmp31 - tmp33;
                auto tmp36 = tmp15 * tmp35;
                auto tmp37 = tmp34 * tmp36;
                auto tmp38 = tmp25 + tmp37;
                tmp38.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*(c10::div_floor_integer(x1, 16L)))));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(16.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp7 = tmp3 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    tmp_acc1_vec = tmp_acc1_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*(c10::div_floor_integer(x0, 16L)))));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (384L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(16.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                auto tmp6 = tmp4 - tmp5;
                auto tmp8 = static_cast<float>(0.0078125);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 * tmp9;
                auto tmp12 = tmp11 * tmp11;
                auto tmp13 = tmp10 * tmp12;
                auto tmp14 = tmp6 * tmp13;
                auto tmp15 = tmp3 - tmp14;
                auto tmp17 = tmp16 * tmp9;
                auto tmp18 = tmp15 - tmp17;
                auto tmp20 = tmp11 * tmp19;
                auto tmp21 = tmp18 * tmp20;
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_4 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp20 = tmp16 * tmp19;
                    tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    tmp_acc1_vec = tmp_acc1_vec + tmp20;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(-3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 < tmp2);
                auto tmp4 = static_cast<float>(3.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = to_float_mask(tmp0 <= tmp5);
                auto tmp8 = tmp0 / tmp5;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp7 * tmp11;
                auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                auto tmp14 = static_cast<float>(0.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                auto tmp19 = tmp17 - tmp18;
                auto tmp21 = static_cast<float>(0.0078125);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 * tmp22;
                auto tmp25 = tmp24 * tmp24;
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp19 * tmp26;
                auto tmp28 = tmp16 - tmp27;
                auto tmp30 = tmp29 * tmp22;
                auto tmp31 = tmp28 - tmp30;
                auto tmp33 = tmp24 * tmp32;
                auto tmp34 = tmp31 * tmp33;
                tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_5 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*(c10::div_floor_integer(x1, 16L)))));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(16.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp5 = tmp3 + tmp4;
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*(c10::div_floor_integer(x0, 16L)))));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(16.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.0078125);
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
                tmp23.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (512L*x2) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x0) + (3072L*x3)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (192L*x1) + (3072L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0) + (192L*x3))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (256L*x0)));
                    }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp1 * tmp4;
                        auto tmp6 = tmp2 - tmp5;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_9 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp34 = in_ptr3[static_cast<long>(x0 + (768L*x1))];
                    auto tmp35 = in_ptr4[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp23 = [&]
                    {
                        auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp24;
                    }
                    ;
                    auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = [&]
                    {
                        auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp30;
                    }
                    ;
                    auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                    auto tmp32 = tmp11 ? tmp28 : tmp31;
                    auto tmp33 = tmp4 ? tmp25 : tmp32;
                    auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                    auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                    tmp_acc0 = tmp_acc0 + tmp22;
                    tmp_acc1 = tmp_acc1 + tmp37;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
            {
                auto tmp23 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                auto tmp24 = in_ptr4[static_cast<long>(x1)];
                auto tmp26 = out_ptr1[static_cast<long>(x1)];
                auto tmp29 = in_ptr5[static_cast<long>(x1)];
                auto tmp34 = out_ptr0[static_cast<long>(x1)];
                auto tmp37 = in_ptr6[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(16);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(32);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = tmp8 & tmp10;
                auto tmp12 = [&]
                {
                    auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x1) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x0) % static_cast<long>(16L)))];
                    return tmp13;
                }
                ;
                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                auto tmp15 = tmp0 >= tmp9;
                auto tmp16 = static_cast<long>(64);
                auto tmp17 = tmp0 < tmp16;
                auto tmp18 = [&]
                {
                    auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x1, 64L))) + (6144L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp19;
                }
                ;
                auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                auto tmp21 = tmp11 ? tmp14 : tmp20;
                auto tmp22 = tmp4 ? tmp7 : tmp21;
                auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                auto tmp27 = static_cast<float>(0.0078125);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp39;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_10 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*(c10::div_floor_integer(x1, 16L)))));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(16.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*(c10::div_floor_integer(x0, 16L)))));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(16.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp10 = tmp8 - tmp9;
                auto tmp12 = static_cast<float>(0.0078125);
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
                tmp25.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_11 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp20 = tmp16 * tmp19;
                    tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    tmp_acc1_vec = tmp_acc1_vec + tmp20;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(-3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 < tmp2);
                auto tmp4 = static_cast<float>(3.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = to_float_mask(tmp0 <= tmp5);
                auto tmp8 = tmp0 / tmp5;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp7 * tmp11;
                auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                auto tmp14 = static_cast<float>(0.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                auto tmp19 = tmp17 - tmp18;
                auto tmp21 = static_cast<float>(0.0078125);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 * tmp22;
                auto tmp25 = tmp24 * tmp24;
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp19 * tmp26;
                auto tmp28 = tmp16 - tmp27;
                auto tmp30 = tmp29 * tmp22;
                auto tmp31 = tmp28 - tmp30;
                auto tmp33 = tmp24 * tmp32;
                auto tmp34 = tmp31 * tmp33;
                tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_12 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*(c10::div_floor_integer(x1, 16L)))));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(16.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp13 = tmp9 * tmp12;
                    tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    tmp_acc1_vec = tmp_acc1_vec + tmp13;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*(c10::div_floor_integer(x0, 16L)))));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(16.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp9 = tmp7 + tmp8;
                auto tmp12 = tmp10 - tmp11;
                auto tmp14 = static_cast<float>(0.0078125);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 * tmp15;
                auto tmp18 = tmp17 * tmp17;
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp12 * tmp19;
                auto tmp21 = tmp9 - tmp20;
                auto tmp23 = tmp22 * tmp15;
                auto tmp24 = tmp21 - tmp23;
                auto tmp26 = tmp17 * tmp25;
                auto tmp27 = tmp24 * tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (512L*x2) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x0) + (3072L*x3)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (192L*x1) + (3072L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0) + (192L*x3))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (256L*x0)));
                    }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp1 * tmp4;
                        auto tmp6 = tmp2 - tmp5;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_16 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp34 = in_ptr3[static_cast<long>(x0 + (768L*x1))];
                    auto tmp35 = in_ptr4[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp23 = [&]
                    {
                        auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp24;
                    }
                    ;
                    auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = [&]
                    {
                        auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp30;
                    }
                    ;
                    auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                    auto tmp32 = tmp11 ? tmp28 : tmp31;
                    auto tmp33 = tmp4 ? tmp25 : tmp32;
                    auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                    auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                    tmp_acc0 = tmp_acc0 + tmp22;
                    tmp_acc1 = tmp_acc1 + tmp37;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
            {
                auto tmp23 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                auto tmp24 = in_ptr4[static_cast<long>(x1)];
                auto tmp26 = out_ptr1[static_cast<long>(x1)];
                auto tmp29 = in_ptr5[static_cast<long>(x1)];
                auto tmp34 = out_ptr0[static_cast<long>(x1)];
                auto tmp37 = in_ptr6[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(16);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(32);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = tmp8 & tmp10;
                auto tmp12 = [&]
                {
                    auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x1) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x0) % static_cast<long>(16L)))];
                    return tmp13;
                }
                ;
                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                auto tmp15 = tmp0 >= tmp9;
                auto tmp16 = static_cast<long>(64);
                auto tmp17 = tmp0 < tmp16;
                auto tmp18 = [&]
                {
                    auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x1, 64L))) + (6144L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp19;
                }
                ;
                auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                auto tmp21 = tmp11 ? tmp14 : tmp20;
                auto tmp22 = tmp4 ? tmp7 : tmp21;
                auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                auto tmp27 = static_cast<float>(0.0078125);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp39;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_native_batch_norm_backward_17 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                    auto tmp1 = static_cast<float>(16.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.0078125);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_18 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp20 = tmp16 * tmp19;
                    tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    tmp_acc1_vec = tmp_acc1_vec + tmp20;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(-3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 < tmp2);
                auto tmp4 = static_cast<float>(3.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = to_float_mask(tmp0 <= tmp5);
                auto tmp8 = tmp0 / tmp5;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp7 * tmp11;
                auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                auto tmp14 = static_cast<float>(0.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                auto tmp19 = tmp17 - tmp18;
                auto tmp21 = static_cast<float>(0.0078125);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 * tmp22;
                auto tmp25 = tmp24 * tmp24;
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp19 * tmp26;
                auto tmp28 = tmp16 - tmp27;
                auto tmp30 = tmp29 * tmp22;
                auto tmp31 = tmp28 - tmp30;
                auto tmp33 = tmp24 * tmp32;
                auto tmp34 = tmp31 * tmp33;
                tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_19 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.0078125);
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
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (512L*x2) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x0) + (3072L*x3)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (192L*x1) + (3072L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0) + (192L*x3))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (256L*x0)));
                    }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp1 * tmp4;
                        auto tmp6 = tmp2 - tmp5;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_23 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp34 = in_ptr3[static_cast<long>(x0 + (768L*x1))];
                    auto tmp35 = in_ptr4[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp23 = [&]
                    {
                        auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp24;
                    }
                    ;
                    auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = [&]
                    {
                        auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp30;
                    }
                    ;
                    auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                    auto tmp32 = tmp11 ? tmp28 : tmp31;
                    auto tmp33 = tmp4 ? tmp25 : tmp32;
                    auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                    auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                    tmp_acc0 = tmp_acc0 + tmp22;
                    tmp_acc1 = tmp_acc1 + tmp37;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
            {
                auto tmp23 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                auto tmp24 = in_ptr4[static_cast<long>(x1)];
                auto tmp26 = out_ptr1[static_cast<long>(x1)];
                auto tmp29 = in_ptr5[static_cast<long>(x1)];
                auto tmp34 = out_ptr0[static_cast<long>(x1)];
                auto tmp37 = in_ptr6[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(16);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(32);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = tmp8 & tmp10;
                auto tmp12 = [&]
                {
                    auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x1) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x0) % static_cast<long>(16L)))];
                    return tmp13;
                }
                ;
                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                auto tmp15 = tmp0 >= tmp9;
                auto tmp16 = static_cast<long>(64);
                auto tmp17 = tmp0 < tmp16;
                auto tmp18 = [&]
                {
                    auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x1, 64L))) + (6144L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp19;
                }
                ;
                auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                auto tmp21 = tmp11 ? tmp14 : tmp20;
                auto tmp22 = tmp4 ? tmp7 : tmp21;
                auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                auto tmp27 = static_cast<float>(0.0078125);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp39;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_24 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp7 = tmp5 - tmp6;
                auto tmp9 = static_cast<float>(0.0078125);
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
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_25 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp20 = tmp16 * tmp19;
                    tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    tmp_acc1_vec = tmp_acc1_vec + tmp20;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(-3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 < tmp2);
                auto tmp4 = static_cast<float>(3.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = to_float_mask(tmp0 <= tmp5);
                auto tmp8 = tmp0 / tmp5;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp7 * tmp11;
                auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                auto tmp14 = static_cast<float>(0.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                auto tmp19 = tmp17 - tmp18;
                auto tmp21 = static_cast<float>(0.0078125);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 * tmp22;
                auto tmp25 = tmp24 * tmp24;
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp19 * tmp26;
                auto tmp28 = tmp16 - tmp27;
                auto tmp30 = tmp29 * tmp22;
                auto tmp31 = tmp28 - tmp30;
                auto tmp33 = tmp24 * tmp32;
                auto tmp34 = tmp31 * tmp33;
                tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_26 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp9 = tmp7 - tmp8;
                auto tmp11 = static_cast<float>(0.0078125);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 * tmp12;
                auto tmp15 = tmp14 * tmp14;
                auto tmp16 = tmp13 * tmp15;
                auto tmp17 = tmp9 * tmp16;
                auto tmp18 = tmp6 - tmp17;
                auto tmp20 = tmp19 * tmp12;
                auto tmp21 = tmp18 - tmp20;
                auto tmp23 = tmp14 * tmp22;
                auto tmp24 = tmp21 * tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (6144L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (512L*x2) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_new_zeros_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x0) + (3072L*x3)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (192L*x1) + (3072L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0) + (192L*x3))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (256L*x0)));
                    }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (192L*x2) + (3072L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp1 * tmp4;
                        auto tmp6 = tmp2 - tmp5;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_30 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp34 = in_ptr3[static_cast<long>(x0 + (768L*x1))];
                    auto tmp35 = in_ptr4[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp23 = [&]
                    {
                        auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp24;
                    }
                    ;
                    auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x0) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x0, 64L))) + (3072L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x1) % static_cast<long>(16L)))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = [&]
                    {
                        auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x0, 64L))) + (6144L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                        return tmp30;
                    }
                    ;
                    auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                    auto tmp32 = tmp11 ? tmp28 : tmp31;
                    auto tmp33 = tmp4 ? tmp25 : tmp32;
                    auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                    auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                    tmp_acc0 = tmp_acc0 + tmp22;
                    tmp_acc1 = tmp_acc1 + tmp37;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
            {
                auto tmp23 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                auto tmp24 = in_ptr4[static_cast<long>(x1)];
                auto tmp26 = out_ptr1[static_cast<long>(x1)];
                auto tmp29 = in_ptr5[static_cast<long>(x1)];
                auto tmp34 = out_ptr0[static_cast<long>(x1)];
                auto tmp37 = in_ptr6[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(16);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(32);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = tmp8 & tmp10;
                auto tmp12 = [&]
                {
                    auto tmp13 = in_ptr1[static_cast<long>((-256L) + (16L*(static_cast<long>(x1) % static_cast<long>(64L))) + (256L*(c10::div_floor_integer(x1, 64L))) + (3072L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x0) % static_cast<long>(16L)))];
                    return tmp13;
                }
                ;
                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                auto tmp15 = tmp0 >= tmp9;
                auto tmp16 = static_cast<long>(64);
                auto tmp17 = tmp0 < tmp16;
                auto tmp18 = [&]
                {
                    auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(c10::div_floor_integer(x1, 64L))) + (6144L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                    return tmp19;
                }
                ;
                auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                auto tmp21 = tmp11 ? tmp14 : tmp20;
                auto tmp22 = tmp4 ? tmp7 : tmp21;
                auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                auto tmp27 = static_cast<float>(0.0078125);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp39;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_31 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(49152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp8 = tmp6 + tmp7;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.0078125);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_32 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp20 = tmp16 * tmp19;
                    tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    tmp_acc1_vec = tmp_acc1_vec + tmp20;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(-3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 < tmp2);
                auto tmp4 = static_cast<float>(3.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = to_float_mask(tmp0 <= tmp5);
                auto tmp8 = tmp0 / tmp5;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp7 * tmp11;
                auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                auto tmp14 = static_cast<float>(0.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                auto tmp19 = tmp17 - tmp18;
                auto tmp21 = static_cast<float>(0.0078125);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 * tmp22;
                auto tmp25 = tmp24 * tmp24;
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp19 * tmp26;
                auto tmp28 = tmp16 - tmp27;
                auto tmp30 = tmp29 * tmp22;
                auto tmp31 = tmp28 - tmp30;
                auto tmp33 = tmp24 * tmp32;
                auto tmp34 = tmp31 * tmp33;
                tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_33 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.0078125);
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
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (1024L*x1) + (16384L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (1024L*x1) + (16384L*x0)));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp8 = tmp0 / tmp5;
                            auto tmp9 = static_cast<float>(0.5);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (1024L*x2) + (16384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_new_zeros_sum_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (784L*x2) + (12544L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (16L*x3) + (784L*x2) + (12544L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            tmp_acc0 = tmp_acc0 + tmp2;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (784L*x0) + (12544L*x3)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (16L*x2) + (16L*x2_inner) + (784L*x1) + (12544L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0) + (256L*x3))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (784L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0) + (12544L*x3))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (16L*x2) + (784L*x1) + (12544L*x3))];
                            auto tmp3 = out_ptr0[static_cast<long>(x1 + (16L*x0) + (256L*x3))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                        out_ptr1[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                    }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (784L*x2) + (12544L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (256L*x0))];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp1 * tmp4;
                        auto tmp6 = tmp2 - tmp5;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x1 + (16L*x3) + (784L*x2) + (12544L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (256L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                        auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                        auto tmp6 = static_cast<float>(0.25);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_37 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer((x0 + x0_inner), 16L))) + (4096L*(c10::div_floor_integer(x1, 16L))) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (256L*(c10::div_floor_integer((x1 + x1_inner), 16L))) + (4096L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.0078125);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp22 = in_ptr2[static_cast<long>(x0 + (1280L*x1))];
                        auto tmp23 = in_ptr3[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(80L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((49L*(static_cast<long>(x0) % static_cast<long>(80L))) + (784L*(c10::div_floor_integer(x0, 80L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(80);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-16L) + (64L*(static_cast<long>(x1) % static_cast<long>(49L))) + (3136L*(c10::div_floor_integer(x0, 80L))) + (50176L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(80L)))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr0[static_cast<long>((49L*(static_cast<long>(x0) % static_cast<long>(80L))) + (784L*(c10::div_floor_integer(x0, 80L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp4 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>((-16L) + (64L*(static_cast<long>(x1) % static_cast<long>(49L))) + (3136L*(c10::div_floor_integer(x0, 80L))) + (50176L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(80L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp4 ? tmp17 : tmp20;
                        auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                        auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                        tmp_acc0 = tmp_acc0 + tmp14;
                        tmp_acc1 = tmp_acc1 + tmp25;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1 + (1280L*x0))];
                    auto tmp16 = in_ptr3[static_cast<long>(x1)];
                    auto tmp18 = out_ptr1[static_cast<long>(x1)];
                    auto tmp21 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr0[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(80L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((49L*(static_cast<long>(x1) % static_cast<long>(80L))) + (784L*(c10::div_floor_integer(x1, 80L))) + (12544L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x0) % static_cast<long>(49L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-16L) + (64L*(static_cast<long>(x0) % static_cast<long>(49L))) + (3136L*(c10::div_floor_integer(x1, 80L))) + (50176L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(80L)))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp17 = decltype(tmp15)(tmp15 - tmp16);
                    auto tmp19 = static_cast<float>(0.002551020408163265);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp22 = decltype(tmp21)(tmp21 * tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp17)(tmp17 * tmp23);
                    auto tmp25 = decltype(tmp14)(tmp14 - tmp24);
                    auto tmp27 = decltype(tmp26)(tmp26 * tmp19);
                    auto tmp28 = decltype(tmp25)(tmp25 - tmp27);
                    auto tmp30 = decltype(tmp21)(tmp21 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    out_ptr3[static_cast<long>(x1 + (1280L*x0))] = tmp31;
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_39 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp38 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 7L)) % static_cast<long>(2L));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = tmp4 == tmp5;
                        auto tmp8 = tmp6 & tmp2;
                        auto tmp7 = [&]
                        {
                            auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x0 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x1, 49L)))), to_float_mask(tmp8));
                            return tmp9;
                        }
                        ;
                        auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = to_float_mask(tmp6);
                        auto tmp13 = at::vec::Vectorized<float>(tmp11);
                        auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = to_float_mask(tmp2);
                    auto tmp18 = at::vec::Vectorized<float>(tmp16);
                    auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp22 = [&]
                    {
                        auto tmp23 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                        auto tmp24 = static_cast<int>(0);
                        auto tmp25 = tmp23 == tmp24;
                        auto tmp27 = tmp25 & tmp2;
                        auto tmp26 = [&]
                        {
                            auto tmp28 = masked_load(in_ptr0 + static_cast<long>(x0 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x1, 49L)))), to_float_mask(tmp27));
                            return tmp28;
                        }
                        ;
                        auto tmp29 = decltype(tmp26())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp26(), to_float_mask(tmp27));
                        auto tmp30 = static_cast<float>(0.0);
                        auto tmp31 = to_float_mask(tmp25);
                        auto tmp32 = at::vec::Vectorized<float>(tmp30);
                        auto tmp33 = decltype(tmp29)::blendv(tmp32, tmp29, tmp31);
                        return tmp33;
                    }
                    ;
                    auto tmp34 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp2));
                    auto tmp35 = decltype(tmp34)::blendv(tmp18, tmp34, tmp17);
                    auto tmp36 = tmp35 + tmp20;
                    auto tmp39 = tmp37 - tmp38;
                    auto tmp40 = tmp36 * tmp39;
                    tmp_acc0_vec = tmp_acc0_vec + tmp21;
                    tmp_acc1_vec = tmp_acc1_vec + tmp40;
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
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp34 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(49L)), 7L)) % static_cast<long>(2L));
                auto tmp1 = static_cast<int>(0);
                auto tmp2 = tmp0 == tmp1;
                auto tmp3 = [&]
                {
                    auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x0) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 == tmp5;
                    auto tmp8 = tmp6 & tmp2;
                    auto tmp7 = [&]
                    {
                        auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x1 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x0) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x0, 49L)))), to_float_mask(tmp8));
                        return tmp9;
                    }
                    ;
                    auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                    auto tmp11 = static_cast<float>(0.0);
                    auto tmp12 = to_float_mask(tmp6);
                    auto tmp13 = at::vec::Vectorized<float>(tmp11);
                    auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                    return tmp14;
                }
                ;
                auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                auto tmp16 = static_cast<float>(0.0);
                auto tmp17 = to_float_mask(tmp2);
                auto tmp18 = at::vec::Vectorized<float>(tmp16);
                auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                auto tmp21 = tmp19 + tmp20;
                auto tmp24 = tmp22 - tmp23;
                auto tmp26 = static_cast<float>(0.002551020408163265);
                auto tmp27 = at::vec::Vectorized<float>(tmp26);
                auto tmp28 = tmp25 * tmp27;
                auto tmp30 = tmp29 * tmp29;
                auto tmp31 = tmp28 * tmp30;
                auto tmp32 = tmp24 * tmp31;
                auto tmp33 = tmp21 - tmp32;
                auto tmp35 = tmp34 * tmp27;
                auto tmp36 = tmp33 - tmp35;
                auto tmp38 = tmp29 * tmp37;
                auto tmp39 = tmp36 * tmp38;
                tmp39.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_40 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_41 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 7L)) % static_cast<long>(2L));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = tmp4 == tmp5;
                        auto tmp8 = tmp6 & tmp2;
                        auto tmp7 = [&]
                        {
                            auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x0 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x1, 49L)))), to_float_mask(tmp8));
                            return tmp9;
                        }
                        ;
                        auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = to_float_mask(tmp6);
                        auto tmp13 = at::vec::Vectorized<float>(tmp11);
                        auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = to_float_mask(tmp2);
                    auto tmp18 = at::vec::Vectorized<float>(tmp16);
                    auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    auto tmp24 = [&]
                    {
                        auto tmp25 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                        auto tmp26 = static_cast<int>(0);
                        auto tmp27 = tmp25 == tmp26;
                        auto tmp29 = tmp27 & tmp2;
                        auto tmp28 = [&]
                        {
                            auto tmp30 = masked_load(in_ptr0 + static_cast<long>(x0 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x1, 49L)))), to_float_mask(tmp29));
                            return tmp30;
                        }
                        ;
                        auto tmp31 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp29));
                        auto tmp32 = static_cast<float>(0.0);
                        auto tmp33 = to_float_mask(tmp27);
                        auto tmp34 = at::vec::Vectorized<float>(tmp32);
                        auto tmp35 = decltype(tmp31)::blendv(tmp34, tmp31, tmp33);
                        return tmp35;
                    }
                    ;
                    auto tmp36 = decltype(tmp24())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp24(), to_float_mask(tmp2));
                    auto tmp37 = decltype(tmp36)::blendv(tmp18, tmp36, tmp17);
                    auto tmp38 = tmp37 + tmp20;
                    auto tmp39 = tmp38 + tmp22;
                    auto tmp42 = tmp40 - tmp41;
                    auto tmp43 = tmp39 * tmp42;
                    tmp_acc0_vec = tmp_acc0_vec + tmp23;
                    tmp_acc1_vec = tmp_acc1_vec + tmp43;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp39 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(49L)), 7L)) % static_cast<long>(2L));
                auto tmp1 = static_cast<int>(0);
                auto tmp2 = tmp0 == tmp1;
                auto tmp3 = [&]
                {
                    auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x0) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 == tmp5;
                    auto tmp8 = tmp6 & tmp2;
                    auto tmp7 = [&]
                    {
                        auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x1 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x0) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x0, 49L)))), to_float_mask(tmp8));
                        return tmp9;
                    }
                    ;
                    auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                    auto tmp11 = static_cast<float>(0.0);
                    auto tmp12 = to_float_mask(tmp6);
                    auto tmp13 = at::vec::Vectorized<float>(tmp11);
                    auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                    return tmp14;
                }
                ;
                auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                auto tmp16 = static_cast<float>(0.0);
                auto tmp17 = to_float_mask(tmp2);
                auto tmp18 = at::vec::Vectorized<float>(tmp16);
                auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                auto tmp21 = tmp19 + tmp20;
                auto tmp23 = tmp21 + tmp22;
                auto tmp26 = tmp24 - tmp25;
                auto tmp28 = static_cast<float>(0.002551020408163265);
                auto tmp29 = at::vec::Vectorized<float>(tmp28);
                auto tmp30 = tmp27 * tmp29;
                auto tmp32 = tmp31 * tmp31;
                auto tmp33 = tmp30 * tmp32;
                auto tmp34 = tmp26 * tmp33;
                auto tmp35 = tmp23 - tmp34;
                auto tmp37 = tmp36 * tmp29;
                auto tmp38 = tmp35 - tmp37;
                auto tmp40 = tmp31 * tmp39;
                auto tmp41 = tmp38 * tmp40;
                tmp41.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
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
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (12544L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (392L*x1) + (19208L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (2401L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (8L*x2) + (392L*x1) + (19208L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (49L*x1) + (2401L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
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
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (512L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (512L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x1) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x0) % static_cast<long>(49L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.002551020408163265);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (512L*x0))] = tmp39;
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


cpp_fused_native_batch_norm_backward_46 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp43 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp44 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 7L)) % static_cast<long>(2L));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = tmp4 == tmp5;
                        auto tmp8 = tmp6 & tmp2;
                        auto tmp7 = [&]
                        {
                            auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x0 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x1, 49L)))), to_float_mask(tmp8));
                            return tmp9;
                        }
                        ;
                        auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = to_float_mask(tmp6);
                        auto tmp13 = at::vec::Vectorized<float>(tmp11);
                        auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = to_float_mask(tmp2);
                    auto tmp18 = at::vec::Vectorized<float>(tmp16);
                    auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                        auto tmp28 = static_cast<int>(0);
                        auto tmp29 = tmp27 == tmp28;
                        auto tmp31 = tmp29 & tmp2;
                        auto tmp30 = [&]
                        {
                            auto tmp32 = masked_load(in_ptr0 + static_cast<long>(x0 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x1, 49L)))), to_float_mask(tmp31));
                            return tmp32;
                        }
                        ;
                        auto tmp33 = decltype(tmp30())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp30(), to_float_mask(tmp31));
                        auto tmp34 = static_cast<float>(0.0);
                        auto tmp35 = to_float_mask(tmp29);
                        auto tmp36 = at::vec::Vectorized<float>(tmp34);
                        auto tmp37 = decltype(tmp33)::blendv(tmp36, tmp33, tmp35);
                        return tmp37;
                    }
                    ;
                    auto tmp38 = decltype(tmp26())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp26(), to_float_mask(tmp2));
                    auto tmp39 = decltype(tmp38)::blendv(tmp18, tmp38, tmp17);
                    auto tmp40 = tmp39 + tmp20;
                    auto tmp41 = tmp40 + tmp22;
                    auto tmp42 = tmp41 + tmp24;
                    auto tmp45 = tmp43 - tmp44;
                    auto tmp46 = tmp42 * tmp45;
                    tmp_acc0_vec = tmp_acc0_vec + tmp25;
                    tmp_acc1_vec = tmp_acc1_vec + tmp46;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp38 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(49L)), 7L)) % static_cast<long>(2L));
                auto tmp1 = static_cast<int>(0);
                auto tmp2 = tmp0 == tmp1;
                auto tmp3 = [&]
                {
                    auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x0) % static_cast<long>(49L))) % static_cast<long>(7L))) % static_cast<long>(2L));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 == tmp5;
                    auto tmp8 = tmp6 & tmp2;
                    auto tmp7 = [&]
                    {
                        auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x1 + (256L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x0) % static_cast<long>(49L))) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(49L)), 14L))) + (4096L*(c10::div_floor_integer(x0, 49L)))), to_float_mask(tmp8));
                        return tmp9;
                    }
                    ;
                    auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                    auto tmp11 = static_cast<float>(0.0);
                    auto tmp12 = to_float_mask(tmp6);
                    auto tmp13 = at::vec::Vectorized<float>(tmp11);
                    auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                    return tmp14;
                }
                ;
                auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                auto tmp16 = static_cast<float>(0.0);
                auto tmp17 = to_float_mask(tmp2);
                auto tmp18 = at::vec::Vectorized<float>(tmp16);
                auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                auto tmp21 = tmp19 + tmp20;
                auto tmp23 = tmp21 + tmp22;
                auto tmp25 = tmp23 + tmp24;
                auto tmp28 = tmp26 - tmp27;
                auto tmp30 = static_cast<float>(0.002551020408163265);
                auto tmp31 = at::vec::Vectorized<float>(tmp30);
                auto tmp32 = tmp29 * tmp31;
                auto tmp34 = tmp33 * tmp33;
                auto tmp35 = tmp32 * tmp34;
                auto tmp36 = tmp28 * tmp35;
                auto tmp37 = tmp25 - tmp36;
                auto tmp39 = tmp38 * tmp31;
                auto tmp40 = tmp37 - tmp39;
                auto tmp42 = tmp33 * tmp41;
                auto tmp43 = tmp40 * tmp42;
                tmp43.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_48 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer(x1, 7L)) % static_cast<long>(2L));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>(x1) % static_cast<long>(7L))) % static_cast<long>(2L));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = tmp4 == tmp5;
                        auto tmp8 = tmp6 & tmp2;
                        auto tmp7 = [&]
                        {
                            auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x2 + (256L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(7L)), 2L))) + (1024L*(c10::div_floor_integer(x1, 14L))) + (4096L*x0)), to_float_mask(tmp8));
                            return tmp9;
                        }
                        ;
                        auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = to_float_mask(tmp6);
                        auto tmp13 = at::vec::Vectorized<float>(tmp11);
                        auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = to_float_mask(tmp2);
                    auto tmp18 = at::vec::Vectorized<float>(tmp16);
                    auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp27 = tmp25 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.002551020408163265);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (12544L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (392L*x1) + (19208L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (2401L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (8L*x2) + (392L*x1) + (19208L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (49L*x1) + (2401L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
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
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (512L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (512L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x1) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x0) % static_cast<long>(49L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.002551020408163265);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (512L*x0))] = tmp39;
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


cpp_fused_native_batch_norm_backward_53 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.002551020408163265);
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
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_54 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_55 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp7 = tmp5 - tmp6;
                auto tmp9 = static_cast<float>(0.002551020408163265);
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
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
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
''')


cpp_fused_clone_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (12544L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (392L*x1) + (19208L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (2401L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (8L*x2) + (392L*x1) + (19208L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (49L*x1) + (2401L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
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
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (512L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (512L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x1) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x0) % static_cast<long>(49L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.002551020408163265);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (512L*x0))] = tmp39;
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


cpp_fused_native_batch_norm_backward_60 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp9 = tmp7 - tmp8;
                auto tmp11 = static_cast<float>(0.002551020408163265);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 * tmp12;
                auto tmp15 = tmp14 * tmp14;
                auto tmp16 = tmp13 * tmp15;
                auto tmp17 = tmp9 * tmp16;
                auto tmp18 = tmp6 - tmp17;
                auto tmp20 = tmp19 * tmp12;
                auto tmp21 = tmp18 - tmp20;
                auto tmp23 = tmp14 * tmp22;
                auto tmp24 = tmp21 * tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_61 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_62 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp8 = tmp6 + tmp7;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.002551020408163265);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (12544L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_new_zeros_sum_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (392L*x1) + (19208L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (2401L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (2401L*x0) + (19208L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (8L*x2) + (392L*x1) + (19208L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (49L*x1) + (2401L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
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
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (392L*x2) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (8L*x3) + (392L*x2) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (512L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x0) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x0, 64L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x1) % static_cast<long>(49L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (512L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-784L) + (49L*(static_cast<long>(x1) % static_cast<long>(64L))) + (784L*(c10::div_floor_integer(x1, 64L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x0) % static_cast<long>(49L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(49L))) + (1568L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.002551020408163265);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (512L*x0))] = tmp39;
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


cpp_fused_native_batch_norm_backward_67 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.002551020408163265);
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
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_68 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_69 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp7 = tmp5 - tmp6;
                auto tmp9 = static_cast<float>(0.002551020408163265);
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
                tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (25088L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (512L*x1) + (25088L*x0)));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp8 = tmp0 / tmp5;
                            auto tmp9 = static_cast<float>(0.5);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (3136L*x2) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_new_zeros_sum_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (1568L*x2) + (76832L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (8L*x3) + (1568L*x2) + (76832L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (9604L*x0) + (76832L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (1568L*x1) + (76832L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (9604L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (9604L*x0) + (76832L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (8L*x2) + (1568L*x1) + (76832L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (49L*x0) + (392L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (9604L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
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
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (8L*x3) + (8L*x3_inner) + (1568L*x2) + (76832L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (8L*x3) + (1568L*x2) + (76832L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_73 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer((x0 + x0_inner), 16L))) + (6272L*(c10::div_floor_integer(x1, 49L))) + (static_cast<long>((x0 + x0_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(49L))) + (784L*(c10::div_floor_integer((x1 + x1_inner), 16L))) + (6272L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.002551020408163265);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp22 = in_ptr2[static_cast<long>(x0 + (640L*x1))];
                        auto tmp23 = in_ptr3[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(80L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((196L*(static_cast<long>(x0) % static_cast<long>(80L))) + (3136L*(c10::div_floor_integer(x0, 80L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(80);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-16L) + (64L*(static_cast<long>(x1) % static_cast<long>(196L))) + (12544L*(c10::div_floor_integer(x0, 80L))) + (100352L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(80L)))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr0[static_cast<long>((196L*(static_cast<long>(x0) % static_cast<long>(80L))) + (3136L*(c10::div_floor_integer(x0, 80L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp4 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>((-16L) + (64L*(static_cast<long>(x1) % static_cast<long>(196L))) + (12544L*(c10::div_floor_integer(x0, 80L))) + (100352L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(80L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp4 ? tmp17 : tmp20;
                        auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                        auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                        tmp_acc0 = tmp_acc0 + tmp14;
                        tmp_acc1 = tmp_acc1 + tmp25;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1 + (640L*x0))];
                    auto tmp16 = in_ptr3[static_cast<long>(x1)];
                    auto tmp18 = out_ptr1[static_cast<long>(x1)];
                    auto tmp21 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr0[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(80L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((196L*(static_cast<long>(x1) % static_cast<long>(80L))) + (3136L*(c10::div_floor_integer(x1, 80L))) + (25088L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-16L) + (64L*(static_cast<long>(x0) % static_cast<long>(196L))) + (12544L*(c10::div_floor_integer(x1, 80L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(80L)))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp17 = decltype(tmp15)(tmp15 - tmp16);
                    auto tmp19 = static_cast<float>(0.0006377551020408163);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp22 = decltype(tmp21)(tmp21 * tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp17)(tmp17 * tmp23);
                    auto tmp25 = decltype(tmp14)(tmp14 - tmp24);
                    auto tmp27 = decltype(tmp26)(tmp26 * tmp19);
                    auto tmp28 = decltype(tmp25)(tmp25 - tmp27);
                    auto tmp30 = decltype(tmp21)(tmp21 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    out_ptr3[static_cast<long>(x1 + (640L*x0))] = tmp31;
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_75 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp38 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 14L)) % static_cast<long>(2L));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                            auto tmp5 = static_cast<int>(0);
                            auto tmp6 = tmp4 == tmp5;
                            auto tmp8 = tmp6 & tmp2;
                            auto tmp7 = [&]
                            {
                                auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x0 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x1, 196L)))), to_float_mask(tmp8));
                                return tmp9;
                            }
                            ;
                            auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                            auto tmp11 = static_cast<float>(0.0);
                            auto tmp12 = to_float_mask(tmp6);
                            auto tmp13 = at::vec::Vectorized<float>(tmp11);
                            auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = to_float_mask(tmp2);
                        auto tmp18 = at::vec::Vectorized<float>(tmp16);
                        auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                        auto tmp21 = tmp19 + tmp20;
                        auto tmp22 = [&]
                        {
                            auto tmp23 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                            auto tmp24 = static_cast<int>(0);
                            auto tmp25 = tmp23 == tmp24;
                            auto tmp27 = tmp25 & tmp2;
                            auto tmp26 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr0 + static_cast<long>(x0 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x1, 196L)))), to_float_mask(tmp27));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp26())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp26(), to_float_mask(tmp27));
                            auto tmp30 = static_cast<float>(0.0);
                            auto tmp31 = to_float_mask(tmp25);
                            auto tmp32 = at::vec::Vectorized<float>(tmp30);
                            auto tmp33 = decltype(tmp29)::blendv(tmp32, tmp29, tmp31);
                            return tmp33;
                        }
                        ;
                        auto tmp34 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp2));
                        auto tmp35 = decltype(tmp34)::blendv(tmp18, tmp34, tmp17);
                        auto tmp36 = tmp35 + tmp20;
                        auto tmp39 = tmp37 - tmp38;
                        auto tmp40 = tmp36 * tmp39;
                        tmp_acc0_vec = tmp_acc0_vec + tmp21;
                        tmp_acc1_vec = tmp_acc1_vec + tmp40;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L)) % static_cast<long>(2L));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = tmp4 == tmp5;
                        auto tmp8 = tmp6 & tmp2;
                        auto tmp7 = [&]
                        {
                            auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x0, 196L)))), to_float_mask(tmp8));
                            return tmp9;
                        }
                        ;
                        auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = to_float_mask(tmp6);
                        auto tmp13 = at::vec::Vectorized<float>(tmp11);
                        auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = to_float_mask(tmp2);
                    auto tmp18 = at::vec::Vectorized<float>(tmp16);
                    auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp24 = tmp22 - tmp23;
                    auto tmp26 = static_cast<float>(0.0006377551020408163);
                    auto tmp27 = at::vec::Vectorized<float>(tmp26);
                    auto tmp28 = tmp25 * tmp27;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp24 * tmp31;
                    auto tmp33 = tmp21 - tmp32;
                    auto tmp35 = tmp34 * tmp27;
                    auto tmp36 = tmp33 - tmp35;
                    auto tmp38 = tmp29 * tmp37;
                    auto tmp39 = tmp36 * tmp38;
                    tmp39.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_76 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_77 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 14L)) % static_cast<long>(2L));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                            auto tmp5 = static_cast<int>(0);
                            auto tmp6 = tmp4 == tmp5;
                            auto tmp8 = tmp6 & tmp2;
                            auto tmp7 = [&]
                            {
                                auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x0 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x1, 196L)))), to_float_mask(tmp8));
                                return tmp9;
                            }
                            ;
                            auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                            auto tmp11 = static_cast<float>(0.0);
                            auto tmp12 = to_float_mask(tmp6);
                            auto tmp13 = at::vec::Vectorized<float>(tmp11);
                            auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = to_float_mask(tmp2);
                        auto tmp18 = at::vec::Vectorized<float>(tmp16);
                        auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                        auto tmp21 = tmp19 + tmp20;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = [&]
                        {
                            auto tmp25 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                            auto tmp26 = static_cast<int>(0);
                            auto tmp27 = tmp25 == tmp26;
                            auto tmp29 = tmp27 & tmp2;
                            auto tmp28 = [&]
                            {
                                auto tmp30 = masked_load(in_ptr0 + static_cast<long>(x0 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x1, 196L)))), to_float_mask(tmp29));
                                return tmp30;
                            }
                            ;
                            auto tmp31 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp29));
                            auto tmp32 = static_cast<float>(0.0);
                            auto tmp33 = to_float_mask(tmp27);
                            auto tmp34 = at::vec::Vectorized<float>(tmp32);
                            auto tmp35 = decltype(tmp31)::blendv(tmp34, tmp31, tmp33);
                            return tmp35;
                        }
                        ;
                        auto tmp36 = decltype(tmp24())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp24(), to_float_mask(tmp2));
                        auto tmp37 = decltype(tmp36)::blendv(tmp18, tmp36, tmp17);
                        auto tmp38 = tmp37 + tmp20;
                        auto tmp39 = tmp38 + tmp22;
                        auto tmp42 = tmp40 - tmp41;
                        auto tmp43 = tmp39 * tmp42;
                        tmp_acc0_vec = tmp_acc0_vec + tmp23;
                        tmp_acc1_vec = tmp_acc1_vec + tmp43;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp39 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L)) % static_cast<long>(2L));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = tmp4 == tmp5;
                        auto tmp8 = tmp6 & tmp2;
                        auto tmp7 = [&]
                        {
                            auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x0, 196L)))), to_float_mask(tmp8));
                            return tmp9;
                        }
                        ;
                        auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = to_float_mask(tmp6);
                        auto tmp13 = at::vec::Vectorized<float>(tmp11);
                        auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = to_float_mask(tmp2);
                    auto tmp18 = at::vec::Vectorized<float>(tmp16);
                    auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = static_cast<float>(0.0006377551020408163);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp27 * tmp29;
                    auto tmp32 = tmp31 * tmp31;
                    auto tmp33 = tmp30 * tmp32;
                    auto tmp34 = tmp26 * tmp33;
                    auto tmp35 = tmp23 - tmp34;
                    auto tmp37 = tmp36 * tmp29;
                    auto tmp38 = tmp35 - tmp37;
                    auto tmp40 = tmp31 * tmp39;
                    auto tmp41 = tmp38 * tmp40;
                    tmp41.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp8 = tmp0 / tmp5;
                            auto tmp9 = static_cast<float>(0.5);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_79 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (4L*x2) + (4L*x2_inner) + (784L*x1) + (153664L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (38416L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (4L*x2) + (784L*x1) + (153664L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (38416L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (256L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (256L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x1) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x1, 64L))) + (25088L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.0006377551020408163);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (256L*x0))] = tmp39;
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


cpp_fused_native_batch_norm_backward_82 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp43 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp44 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 14L)) % static_cast<long>(2L));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                            auto tmp5 = static_cast<int>(0);
                            auto tmp6 = tmp4 == tmp5;
                            auto tmp8 = tmp6 & tmp2;
                            auto tmp7 = [&]
                            {
                                auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x0 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x1, 196L)))), to_float_mask(tmp8));
                                return tmp9;
                            }
                            ;
                            auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                            auto tmp11 = static_cast<float>(0.0);
                            auto tmp12 = to_float_mask(tmp6);
                            auto tmp13 = at::vec::Vectorized<float>(tmp11);
                            auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = to_float_mask(tmp2);
                        auto tmp18 = at::vec::Vectorized<float>(tmp16);
                        auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                        auto tmp21 = tmp19 + tmp20;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                            auto tmp28 = static_cast<int>(0);
                            auto tmp29 = tmp27 == tmp28;
                            auto tmp31 = tmp29 & tmp2;
                            auto tmp30 = [&]
                            {
                                auto tmp32 = masked_load(in_ptr0 + static_cast<long>(x0 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x1) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x1, 196L)))), to_float_mask(tmp31));
                                return tmp32;
                            }
                            ;
                            auto tmp33 = decltype(tmp30())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp30(), to_float_mask(tmp31));
                            auto tmp34 = static_cast<float>(0.0);
                            auto tmp35 = to_float_mask(tmp29);
                            auto tmp36 = at::vec::Vectorized<float>(tmp34);
                            auto tmp37 = decltype(tmp33)::blendv(tmp36, tmp33, tmp35);
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp26())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp26(), to_float_mask(tmp2));
                        auto tmp39 = decltype(tmp38)::blendv(tmp18, tmp38, tmp17);
                        auto tmp40 = tmp39 + tmp20;
                        auto tmp41 = tmp40 + tmp22;
                        auto tmp42 = tmp41 + tmp24;
                        auto tmp45 = tmp43 - tmp44;
                        auto tmp46 = tmp42 * tmp45;
                        tmp_acc0_vec = tmp_acc0_vec + tmp25;
                        tmp_acc1_vec = tmp_acc1_vec + tmp46;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp38 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L)) % static_cast<long>(2L));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) % static_cast<long>(2L));
                        auto tmp5 = static_cast<int>(0);
                        auto tmp6 = tmp4 == tmp5;
                        auto tmp8 = tmp6 & tmp2;
                        auto tmp7 = [&]
                        {
                            auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*(c10::div_floor_integer((static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 28L))) + (6272L*(c10::div_floor_integer(x0, 196L)))), to_float_mask(tmp8));
                            return tmp9;
                        }
                        ;
                        auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = to_float_mask(tmp6);
                        auto tmp13 = at::vec::Vectorized<float>(tmp11);
                        auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = to_float_mask(tmp2);
                    auto tmp18 = at::vec::Vectorized<float>(tmp16);
                    auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp28 = tmp26 - tmp27;
                    auto tmp30 = static_cast<float>(0.0006377551020408163);
                    auto tmp31 = at::vec::Vectorized<float>(tmp30);
                    auto tmp32 = tmp29 * tmp31;
                    auto tmp34 = tmp33 * tmp33;
                    auto tmp35 = tmp32 * tmp34;
                    auto tmp36 = tmp28 * tmp35;
                    auto tmp37 = tmp25 - tmp36;
                    auto tmp39 = tmp38 * tmp31;
                    auto tmp40 = tmp37 - tmp39;
                    auto tmp42 = tmp33 * tmp41;
                    auto tmp43 = tmp40 * tmp42;
                    tmp43.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_83 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_84 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                        auto tmp0 = c10::convert<int>(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(2L));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<int>(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(2L));
                            auto tmp5 = static_cast<int>(0);
                            auto tmp6 = tmp4 == tmp5;
                            auto tmp8 = tmp6 & tmp2;
                            auto tmp7 = [&]
                            {
                                auto tmp9 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 2L))) + (896L*(c10::div_floor_integer(x1, 28L))) + (6272L*x0)), to_float_mask(tmp8));
                                return tmp9;
                            }
                            ;
                            auto tmp10 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp8));
                            auto tmp11 = static_cast<float>(0.0);
                            auto tmp12 = to_float_mask(tmp6);
                            auto tmp13 = at::vec::Vectorized<float>(tmp11);
                            auto tmp14 = decltype(tmp10)::blendv(tmp13, tmp10, tmp12);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = to_float_mask(tmp2);
                        auto tmp18 = at::vec::Vectorized<float>(tmp16);
                        auto tmp19 = decltype(tmp15)::blendv(tmp18, tmp15, tmp17);
                        auto tmp21 = tmp19 + tmp20;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp25 = tmp23 + tmp24;
                        auto tmp27 = tmp25 + tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0006377551020408163);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp8 = tmp0 / tmp5;
                            auto tmp9 = static_cast<float>(0.5);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_86 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (4L*x2) + (4L*x2_inner) + (784L*x1) + (153664L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (38416L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (4L*x2) + (784L*x1) + (153664L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (38416L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (256L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (256L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x1) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x1, 64L))) + (25088L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.0006377551020408163);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (256L*x0))] = tmp39;
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


cpp_fused_native_batch_norm_backward_89 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
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


cpp_fused_native_batch_norm_backward_90 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_91 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.0006377551020408163);
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
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp8 = tmp0 / tmp5;
                            auto tmp9 = static_cast<float>(0.5);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_index_put_new_zeros_sum_93 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (4L*x2) + (4L*x2_inner) + (784L*x1) + (153664L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (38416L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (4L*x2) + (784L*x1) + (153664L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (38416L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (256L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (256L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x1) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x1, 64L))) + (25088L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.0006377551020408163);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (256L*x0))] = tmp39;
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


cpp_fused_native_batch_norm_backward_96 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp11 = static_cast<float>(0.0006377551020408163);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp6 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_97 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_98 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0006377551020408163);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (25088L*x0)));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp8 = tmp0 / tmp5;
                            auto tmp9 = static_cast<float>(0.5);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 + tmp10;
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_new_zeros_sum_100 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (4L*x2) + (4L*x2_inner) + (784L*x1) + (153664L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (38416L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (38416L*x0) + (153664L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (4L*x2) + (784L*x1) + (153664L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0) + (784L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (38416L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (784L*x2) + (153664L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp1 = in_ptr0[static_cast<long>(x1 + (4L*x3) + (784L*x2) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.25);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp34 = in_ptr3[static_cast<long>(x0 + (256L*x1))];
                        auto tmp35 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(static_cast<long>(x0) % static_cast<long>(64L));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(64);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp4 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x0) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x0, 64L))) + (12544L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x1) % static_cast<long>(196L)))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = [&]
                        {
                            auto tmp30 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x1) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x0, 64L))) + (25088L*(c10::div_floor_integer(x1, 196L))) + (static_cast<long>(x0) % static_cast<long>(64L)))];
                            return tmp30;
                        }
                        ;
                        auto tmp31 = tmp15 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                        auto tmp32 = tmp11 ? tmp28 : tmp31;
                        auto tmp33 = tmp4 ? tmp25 : tmp32;
                        auto tmp36 = decltype(tmp34)(tmp34 - tmp35);
                        auto tmp37 = decltype(tmp33)(tmp33 * tmp36);
                        tmp_acc0 = tmp_acc0 + tmp22;
                        tmp_acc1 = tmp_acc1 + tmp37;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1 + (256L*x0))];
                    auto tmp24 = in_ptr4[static_cast<long>(x1)];
                    auto tmp26 = out_ptr1[static_cast<long>(x1)];
                    auto tmp29 = in_ptr5[static_cast<long>(x1)];
                    auto tmp34 = out_ptr0[static_cast<long>(x1)];
                    auto tmp37 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(static_cast<long>(x1) % static_cast<long>(64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(196L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(32);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-3136L) + (196L*(static_cast<long>(x1) % static_cast<long>(64L))) + (3136L*(c10::div_floor_integer(x1, 64L))) + (12544L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-32L) + (32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer(x1, 64L))) + (25088L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp27 = static_cast<float>(0.0006377551020408163);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                    auto tmp32 = decltype(tmp25)(tmp25 * tmp31);
                    auto tmp33 = decltype(tmp22)(tmp22 - tmp32);
                    auto tmp35 = decltype(tmp34)(tmp34 * tmp27);
                    auto tmp36 = decltype(tmp33)(tmp33 - tmp35);
                    auto tmp38 = decltype(tmp29)(tmp29 * tmp37);
                    auto tmp39 = decltype(tmp36)(tmp36 * tmp38);
                    out_ptr2[static_cast<long>(x1 + (256L*x0))] = tmp39;
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


cpp_fused_convolution_backward_native_batch_norm_backward_103 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_104 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_105 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(9.964923469387754e-06);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_79, primals_82, primals_85, primals_88, primals_91, primals_94, primals_97, primals_100, primals_103, primals_106, primals_109, primals_112, primals_115, primals_118, primals_121, primals_124, primals_127, primals_130, primals_133, primals_136, primals_139, primals_142, primals_145, primals_148, primals_151, primals_154, primals_157, primals_160, primals_163, primals_166, primals_169, primals_172, primals_175, primals_178, primals_181, primals_184, primals_187, primals_190, primals_193, primals_196, primals_199, primals_201, primals_205, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_415, convolution, squeeze_1, add_4, div, convolution_1, squeeze_4, add_10, div_1, convolution_2, squeeze_7, add_16, div_2, convolution_3, squeeze_10, view_1, mm, squeeze_13, view_12, view_13, mm_1, squeeze_16, view_17, mm_2, squeeze_19, view_20, view_21, mm_3, squeeze_22, view_25, mm_4, squeeze_25, view_36, view_37, mm_5, squeeze_28, view_41, mm_6, squeeze_31, view_44, view_45, mm_7, squeeze_34, view_49, mm_8, squeeze_37, view_60, view_61, mm_9, squeeze_40, view_65, mm_10, squeeze_43, view_68, view_69, mm_11, squeeze_46, view_73, mm_12, squeeze_49, view_84, view_85, mm_13, squeeze_52, view_89, mm_14, squeeze_55, view_92, view_93, mm_15, squeeze_58, view_97, mm_16, squeeze_61, view_104, mm_17, squeeze_64, view_115, view_116, mm_18, squeeze_67, view_120, mm_19, squeeze_70, view_123, view_124, mm_20, squeeze_73, view_128, mm_21, squeeze_76, view_139, view_140, mm_22, squeeze_79, view_144, mm_23, squeeze_82, view_147, view_148, mm_24, squeeze_85, view_152, mm_25, squeeze_88, view_163, view_164, mm_26, squeeze_91, view_168, mm_27, squeeze_94, view_171, view_172, mm_28, squeeze_97, view_176, mm_29, squeeze_100, view_187, view_188, mm_30, squeeze_103, view_192, mm_31, squeeze_106, view_195, view_196, mm_32, squeeze_109, view_200, mm_33, squeeze_112, view_211, view_212, mm_34, squeeze_115, view_216, mm_35, squeeze_118, view_219, view_220, mm_36, squeeze_121, view_224, mm_37, squeeze_124, view_231, mm_38, squeeze_127, view_242, view_243, mm_39, squeeze_130, view_247, mm_40, squeeze_133, view_250, view_251, mm_41, squeeze_136, view_255, mm_42, squeeze_139, view_266, view_267, mm_43, squeeze_142, view_271, mm_44, squeeze_145, view_274, view_275, mm_45, squeeze_148, view_279, mm_46, squeeze_151, view_290, view_291, mm_47, squeeze_154, view_295, mm_48, squeeze_157, view_298, view_299, mm_49, squeeze_160, view_303, mm_50, squeeze_163, view_314, view_315, mm_51, squeeze_166, view_319, mm_52, squeeze_169, view_322, view_323, mm_53, squeeze_172, view_327, mm_54, squeeze_175, view_338, view_339, mm_55, squeeze_178, view_343, mm_56, squeeze_181, view_346, view_347, mm_57, squeeze_184, mean, clone_81, clone_82, permute_117, permute_121, unsqueeze_25, permute_127, unsqueeze_29, permute_131, unsqueeze_33, permute_135, permute_138, permute_139, alias_14, permute_140, permute_141, unsqueeze_37, permute_147, unsqueeze_41, permute_151, unsqueeze_45, permute_155, unsqueeze_49, permute_159, permute_162, permute_163, alias_15, permute_164, permute_165, unsqueeze_53, permute_171, unsqueeze_57, permute_175, unsqueeze_61, permute_179, unsqueeze_65, permute_183, permute_186, permute_187, alias_16, permute_188, permute_189, unsqueeze_69, permute_195, unsqueeze_73, permute_199, unsqueeze_77, permute_203, unsqueeze_81, permute_207, permute_210, permute_211, alias_17, permute_212, permute_213, unsqueeze_85, permute_219, unsqueeze_89, permute_223, unsqueeze_93, permute_227, unsqueeze_97, permute_231, permute_234, permute_235, alias_18, permute_236, permute_237, unsqueeze_101, permute_241, unsqueeze_105, permute_247, unsqueeze_109, permute_251, unsqueeze_113, permute_255, unsqueeze_117, permute_259, permute_262, permute_263, alias_19, permute_264, permute_265, unsqueeze_121, permute_271, unsqueeze_125, permute_275, unsqueeze_129, permute_279, unsqueeze_133, permute_283, permute_286, permute_287, alias_20, permute_288, permute_289, unsqueeze_137, permute_295, unsqueeze_141, permute_299, unsqueeze_145, permute_303, unsqueeze_149, permute_307, permute_310, permute_311, alias_21, permute_312, permute_313, unsqueeze_153, permute_319, unsqueeze_157, permute_323, unsqueeze_161, permute_327, unsqueeze_165, permute_331, permute_334, permute_335, alias_22, permute_336, permute_337, unsqueeze_169, permute_343, unsqueeze_173, permute_347, unsqueeze_177, permute_351, unsqueeze_181, permute_355, permute_358, permute_359, alias_23, permute_360, permute_361, unsqueeze_185, permute_365, unsqueeze_189, permute_371, unsqueeze_193, permute_375, unsqueeze_197, permute_379, unsqueeze_201, permute_383, permute_386, permute_387, alias_24, permute_388, permute_389, unsqueeze_205, permute_395, unsqueeze_209, permute_399, unsqueeze_213, permute_403, unsqueeze_217, permute_407, permute_410, permute_411, alias_25, permute_412, permute_413, unsqueeze_221, permute_419, unsqueeze_225, permute_423, unsqueeze_229, permute_427, unsqueeze_233, permute_431, permute_434, permute_435, alias_26, permute_436, permute_437, unsqueeze_237, permute_443, unsqueeze_241, permute_447, unsqueeze_245, permute_451, unsqueeze_249, permute_455, permute_458, permute_459, alias_27, permute_460, permute_461, unsqueeze_253, permute_467, unsqueeze_259, unsqueeze_271, unsqueeze_283, unsqueeze_295, tangents_1 = args
    args.clear()
    assert_size_stride(primals_15, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_16, (16, ), (1, ))
    assert_size_stride(primals_18, (32, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_21, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_76, (640, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_139, (1280, ), (1, ))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_187, (384, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_209, (196, 196), (196, 1))
    assert_size_stride(primals_210, (196, 196), (196, 1))
    assert_size_stride(primals_211, (196, 196), (196, 1))
    assert_size_stride(primals_212, (196, 196), (196, 1))
    assert_size_stride(primals_213, (49, 196), (196, 1))
    assert_size_stride(primals_214, (49, 49), (49, 1))
    assert_size_stride(primals_215, (49, 49), (49, 1))
    assert_size_stride(primals_216, (49, 49), (49, 1))
    assert_size_stride(primals_217, (49, 49), (49, 1))
    assert_size_stride(primals_218, (16, 49), (49, 1))
    assert_size_stride(primals_219, (16, 16), (16, 1))
    assert_size_stride(primals_220, (16, 16), (16, 1))
    assert_size_stride(primals_221, (16, 16), (16, 1))
    assert_size_stride(primals_222, (16, 16), (16, 1))
    assert_size_stride(primals_415, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(add_4, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(add_10, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(div_1, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_2, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(add_16, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(div_2, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_3, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(view_1, (1568, 128), (128, 1))
    assert_size_stride(mm, (1568, 256), (256, 1))
    assert_size_stride(squeeze_13, (256, ), (1, ))
    assert_size_stride(view_12, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_13, (1568, 128), (128, 1))
    assert_size_stride(mm_1, (1568, 128), (128, 1))
    assert_size_stride(squeeze_16, (128, ), (1, ))
    assert_size_stride(view_17, (1568, 128), (128, 1))
    assert_size_stride(mm_2, (1568, 256), (256, 1))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(view_20, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_21, (1568, 256), (256, 1))
    assert_size_stride(mm_3, (1568, 128), (128, 1))
    assert_size_stride(squeeze_22, (128, ), (1, ))
    assert_size_stride(view_25, (1568, 128), (128, 1))
    assert_size_stride(mm_4, (1568, 256), (256, 1))
    assert_size_stride(squeeze_25, (256, ), (1, ))
    assert_size_stride(view_36, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_37, (1568, 128), (128, 1))
    assert_size_stride(mm_5, (1568, 128), (128, 1))
    assert_size_stride(squeeze_28, (128, ), (1, ))
    assert_size_stride(view_41, (1568, 128), (128, 1))
    assert_size_stride(mm_6, (1568, 256), (256, 1))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(view_44, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_45, (1568, 256), (256, 1))
    assert_size_stride(mm_7, (1568, 128), (128, 1))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(view_49, (1568, 128), (128, 1))
    assert_size_stride(mm_8, (1568, 256), (256, 1))
    assert_size_stride(squeeze_37, (256, ), (1, ))
    assert_size_stride(view_60, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_61, (1568, 128), (128, 1))
    assert_size_stride(mm_9, (1568, 128), (128, 1))
    assert_size_stride(squeeze_40, (128, ), (1, ))
    assert_size_stride(view_65, (1568, 128), (128, 1))
    assert_size_stride(mm_10, (1568, 256), (256, 1))
    assert_size_stride(squeeze_43, (256, ), (1, ))
    assert_size_stride(view_68, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_69, (1568, 256), (256, 1))
    assert_size_stride(mm_11, (1568, 128), (128, 1))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(view_73, (1568, 128), (128, 1))
    assert_size_stride(mm_12, (1568, 256), (256, 1))
    assert_size_stride(squeeze_49, (256, ), (1, ))
    assert_size_stride(view_84, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_85, (1568, 128), (128, 1))
    assert_size_stride(mm_13, (1568, 128), (128, 1))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(view_89, (1568, 128), (128, 1))
    assert_size_stride(mm_14, (1568, 256), (256, 1))
    assert_size_stride(squeeze_55, (256, ), (1, ))
    assert_size_stride(view_92, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_93, (1568, 256), (256, 1))
    assert_size_stride(mm_15, (1568, 128), (128, 1))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(view_97, (1568, 128), (128, 1))
    assert_size_stride(mm_16, (1568, 640), (640, 1))
    assert_size_stride(squeeze_61, (640, ), (1, ))
    assert_size_stride(view_104, (392, 128), (128, 1))
    assert_size_stride(mm_17, (392, 128), (128, 1))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(view_115, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_116, (392, 512), (512, 1))
    assert_size_stride(mm_18, (392, 256), (256, 1))
    assert_size_stride(squeeze_67, (256, ), (1, ))
    assert_size_stride(view_120, (392, 256), (256, 1))
    assert_size_stride(mm_19, (392, 512), (512, 1))
    assert_size_stride(squeeze_70, (512, ), (1, ))
    assert_size_stride(view_123, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_124, (392, 512), (512, 1))
    assert_size_stride(mm_20, (392, 256), (256, 1))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(view_128, (392, 256), (256, 1))
    assert_size_stride(mm_21, (392, 512), (512, 1))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(view_139, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_140, (392, 256), (256, 1))
    assert_size_stride(mm_22, (392, 256), (256, 1))
    assert_size_stride(squeeze_79, (256, ), (1, ))
    assert_size_stride(view_144, (392, 256), (256, 1))
    assert_size_stride(mm_23, (392, 512), (512, 1))
    assert_size_stride(squeeze_82, (512, ), (1, ))
    assert_size_stride(view_147, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_148, (392, 512), (512, 1))
    assert_size_stride(mm_24, (392, 256), (256, 1))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(view_152, (392, 256), (256, 1))
    assert_size_stride(mm_25, (392, 512), (512, 1))
    assert_size_stride(squeeze_88, (512, ), (1, ))
    assert_size_stride(view_163, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_164, (392, 256), (256, 1))
    assert_size_stride(mm_26, (392, 256), (256, 1))
    assert_size_stride(squeeze_91, (256, ), (1, ))
    assert_size_stride(view_168, (392, 256), (256, 1))
    assert_size_stride(mm_27, (392, 512), (512, 1))
    assert_size_stride(squeeze_94, (512, ), (1, ))
    assert_size_stride(view_171, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_172, (392, 512), (512, 1))
    assert_size_stride(mm_28, (392, 256), (256, 1))
    assert_size_stride(squeeze_97, (256, ), (1, ))
    assert_size_stride(view_176, (392, 256), (256, 1))
    assert_size_stride(mm_29, (392, 512), (512, 1))
    assert_size_stride(squeeze_100, (512, ), (1, ))
    assert_size_stride(view_187, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_188, (392, 256), (256, 1))
    assert_size_stride(mm_30, (392, 256), (256, 1))
    assert_size_stride(squeeze_103, (256, ), (1, ))
    assert_size_stride(view_192, (392, 256), (256, 1))
    assert_size_stride(mm_31, (392, 512), (512, 1))
    assert_size_stride(squeeze_106, (512, ), (1, ))
    assert_size_stride(view_195, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_196, (392, 512), (512, 1))
    assert_size_stride(mm_32, (392, 256), (256, 1))
    assert_size_stride(squeeze_109, (256, ), (1, ))
    assert_size_stride(view_200, (392, 256), (256, 1))
    assert_size_stride(mm_33, (392, 512), (512, 1))
    assert_size_stride(squeeze_112, (512, ), (1, ))
    assert_size_stride(view_211, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_212, (392, 256), (256, 1))
    assert_size_stride(mm_34, (392, 256), (256, 1))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(view_216, (392, 256), (256, 1))
    assert_size_stride(mm_35, (392, 512), (512, 1))
    assert_size_stride(squeeze_118, (512, ), (1, ))
    assert_size_stride(view_219, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_220, (392, 512), (512, 1))
    assert_size_stride(mm_36, (392, 256), (256, 1))
    assert_size_stride(squeeze_121, (256, ), (1, ))
    assert_size_stride(view_224, (392, 256), (256, 1))
    assert_size_stride(mm_37, (392, 1280), (1280, 1))
    assert_size_stride(squeeze_124, (1280, ), (1, ))
    assert_size_stride(view_231, (128, 256), (256, 1))
    assert_size_stride(mm_38, (128, 256), (256, 1))
    assert_size_stride(squeeze_127, (256, ), (1, ))
    assert_size_stride(view_242, (8, 16, 1024), (16384, 1024, 1))
    assert_size_stride(view_243, (128, 1024), (1024, 1))
    assert_size_stride(mm_39, (128, 384), (384, 1))
    assert_size_stride(squeeze_130, (384, ), (1, ))
    assert_size_stride(view_247, (128, 384), (384, 1))
    assert_size_stride(mm_40, (128, 768), (768, 1))
    assert_size_stride(squeeze_133, (768, ), (1, ))
    assert_size_stride(view_250, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_251, (128, 768), (768, 1))
    assert_size_stride(mm_41, (128, 384), (384, 1))
    assert_size_stride(squeeze_136, (384, ), (1, ))
    assert_size_stride(view_255, (128, 384), (384, 1))
    assert_size_stride(mm_42, (128, 768), (768, 1))
    assert_size_stride(squeeze_139, (768, ), (1, ))
    assert_size_stride(view_266, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_267, (128, 384), (384, 1))
    assert_size_stride(mm_43, (128, 384), (384, 1))
    assert_size_stride(squeeze_142, (384, ), (1, ))
    assert_size_stride(view_271, (128, 384), (384, 1))
    assert_size_stride(mm_44, (128, 768), (768, 1))
    assert_size_stride(squeeze_145, (768, ), (1, ))
    assert_size_stride(view_274, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_275, (128, 768), (768, 1))
    assert_size_stride(mm_45, (128, 384), (384, 1))
    assert_size_stride(squeeze_148, (384, ), (1, ))
    assert_size_stride(view_279, (128, 384), (384, 1))
    assert_size_stride(mm_46, (128, 768), (768, 1))
    assert_size_stride(squeeze_151, (768, ), (1, ))
    assert_size_stride(view_290, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_291, (128, 384), (384, 1))
    assert_size_stride(mm_47, (128, 384), (384, 1))
    assert_size_stride(squeeze_154, (384, ), (1, ))
    assert_size_stride(view_295, (128, 384), (384, 1))
    assert_size_stride(mm_48, (128, 768), (768, 1))
    assert_size_stride(squeeze_157, (768, ), (1, ))
    assert_size_stride(view_298, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_299, (128, 768), (768, 1))
    assert_size_stride(mm_49, (128, 384), (384, 1))
    assert_size_stride(squeeze_160, (384, ), (1, ))
    assert_size_stride(view_303, (128, 384), (384, 1))
    assert_size_stride(mm_50, (128, 768), (768, 1))
    assert_size_stride(squeeze_163, (768, ), (1, ))
    assert_size_stride(view_314, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_315, (128, 384), (384, 1))
    assert_size_stride(mm_51, (128, 384), (384, 1))
    assert_size_stride(squeeze_166, (384, ), (1, ))
    assert_size_stride(view_319, (128, 384), (384, 1))
    assert_size_stride(mm_52, (128, 768), (768, 1))
    assert_size_stride(squeeze_169, (768, ), (1, ))
    assert_size_stride(view_322, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_323, (128, 768), (768, 1))
    assert_size_stride(mm_53, (128, 384), (384, 1))
    assert_size_stride(squeeze_172, (384, ), (1, ))
    assert_size_stride(view_327, (128, 384), (384, 1))
    assert_size_stride(mm_54, (128, 768), (768, 1))
    assert_size_stride(squeeze_175, (768, ), (1, ))
    assert_size_stride(view_338, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_339, (128, 384), (384, 1))
    assert_size_stride(mm_55, (128, 384), (384, 1))
    assert_size_stride(squeeze_178, (384, ), (1, ))
    assert_size_stride(view_343, (128, 384), (384, 1))
    assert_size_stride(mm_56, (128, 768), (768, 1))
    assert_size_stride(squeeze_181, (768, ), (1, ))
    assert_size_stride(view_346, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_347, (128, 768), (768, 1))
    assert_size_stride(mm_57, (128, 384), (384, 1))
    assert_size_stride(squeeze_184, (384, ), (1, ))
    assert_size_stride(mean, (8, 384), (384, 1))
    assert_size_stride(clone_81, (8, 384), (384, 1))
    assert_size_stride(clone_82, (8, 384), (384, 1))
    assert_size_stride(permute_117, (1000, 384), (384, 1))
    assert_size_stride(permute_121, (1000, 384), (384, 1))
    assert_size_stride(unsqueeze_25, (1, 384), (384, 1))
    assert_size_stride(permute_127, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_29, (1, 768), (768, 1))
    assert_size_stride(permute_131, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_33, (1, 384), (384, 1))
    assert_size_stride(permute_135, (384, 384), (384, 1))
    assert_size_stride(permute_138, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_139, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_14, (8, 12, 16, 16), (3072, 1, 192, 12))
    assert_size_stride(permute_140, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_141, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_37, (1, 768), (768, 1))
    assert_size_stride(permute_147, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_41, (1, 384), (384, 1))
    assert_size_stride(permute_151, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_45, (1, 768), (768, 1))
    assert_size_stride(permute_155, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_49, (1, 384), (384, 1))
    assert_size_stride(permute_159, (384, 384), (384, 1))
    assert_size_stride(permute_162, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_163, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_15, (8, 12, 16, 16), (3072, 1, 192, 12))
    assert_size_stride(permute_164, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_165, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_53, (1, 768), (768, 1))
    assert_size_stride(permute_171, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_57, (1, 384), (384, 1))
    assert_size_stride(permute_175, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_61, (1, 768), (768, 1))
    assert_size_stride(permute_179, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_65, (1, 384), (384, 1))
    assert_size_stride(permute_183, (384, 384), (384, 1))
    assert_size_stride(permute_186, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_187, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_16, (8, 12, 16, 16), (3072, 1, 192, 12))
    assert_size_stride(permute_188, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_189, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_69, (1, 768), (768, 1))
    assert_size_stride(permute_195, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_73, (1, 384), (384, 1))
    assert_size_stride(permute_199, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_77, (1, 768), (768, 1))
    assert_size_stride(permute_203, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_81, (1, 384), (384, 1))
    assert_size_stride(permute_207, (384, 384), (384, 1))
    assert_size_stride(permute_210, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_211, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_17, (8, 12, 16, 16), (3072, 1, 192, 12))
    assert_size_stride(permute_212, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_213, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_85, (1, 768), (768, 1))
    assert_size_stride(permute_219, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_89, (1, 384), (384, 1))
    assert_size_stride(permute_223, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_93, (1, 768), (768, 1))
    assert_size_stride(permute_227, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_97, (1, 384), (384, 1))
    assert_size_stride(permute_231, (384, 1024), (1024, 1))
    assert_size_stride(permute_234, (128, 49, 16), (784, 1, 49))
    assert_size_stride(permute_235, (128, 64, 49), (3136, 1, 64))
    assert_size_stride(alias_18, (8, 16, 16, 49), (12544, 1, 784, 16))
    assert_size_stride(permute_236, (128, 16, 16), (256, 1, 16))
    assert_size_stride(permute_237, (128, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_101, (1, 256), (256, 1))
    assert_size_stride(permute_241, (256, 256), (256, 1))
    assert_size_stride(unsqueeze_105, (1, 1280), (1280, 1))
    assert_size_stride(permute_247, (1280, 256), (256, 1))
    assert_size_stride(unsqueeze_109, (1, 256), (256, 1))
    assert_size_stride(permute_251, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_113, (1, 512), (512, 1))
    assert_size_stride(permute_255, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_117, (1, 256), (256, 1))
    assert_size_stride(permute_259, (256, 256), (256, 1))
    assert_size_stride(permute_262, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_263, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_19, (8, 8, 49, 49), (19208, 1, 392, 8))
    assert_size_stride(permute_264, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_265, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_121, (1, 512), (512, 1))
    assert_size_stride(permute_271, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_125, (1, 256), (256, 1))
    assert_size_stride(permute_275, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_129, (1, 512), (512, 1))
    assert_size_stride(permute_279, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_133, (1, 256), (256, 1))
    assert_size_stride(permute_283, (256, 256), (256, 1))
    assert_size_stride(permute_286, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_287, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_20, (8, 8, 49, 49), (19208, 1, 392, 8))
    assert_size_stride(permute_288, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_289, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_137, (1, 512), (512, 1))
    assert_size_stride(permute_295, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_141, (1, 256), (256, 1))
    assert_size_stride(permute_299, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_145, (1, 512), (512, 1))
    assert_size_stride(permute_303, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_149, (1, 256), (256, 1))
    assert_size_stride(permute_307, (256, 256), (256, 1))
    assert_size_stride(permute_310, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_311, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_21, (8, 8, 49, 49), (19208, 1, 392, 8))
    assert_size_stride(permute_312, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_313, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_153, (1, 512), (512, 1))
    assert_size_stride(permute_319, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_157, (1, 256), (256, 1))
    assert_size_stride(permute_323, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_161, (1, 512), (512, 1))
    assert_size_stride(permute_327, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_165, (1, 256), (256, 1))
    assert_size_stride(permute_331, (256, 256), (256, 1))
    assert_size_stride(permute_334, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_335, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_22, (8, 8, 49, 49), (19208, 1, 392, 8))
    assert_size_stride(permute_336, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_337, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_169, (1, 512), (512, 1))
    assert_size_stride(permute_343, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_173, (1, 256), (256, 1))
    assert_size_stride(permute_347, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_177, (1, 512), (512, 1))
    assert_size_stride(permute_351, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_181, (1, 256), (256, 1))
    assert_size_stride(permute_355, (256, 512), (512, 1))
    assert_size_stride(permute_358, (64, 196, 49), (9604, 1, 196))
    assert_size_stride(permute_359, (64, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_23, (8, 8, 49, 196), (76832, 1, 1568, 8))
    assert_size_stride(permute_360, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_361, (64, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_185, (1, 128), (128, 1))
    assert_size_stride(permute_365, (128, 128), (128, 1))
    assert_size_stride(unsqueeze_189, (1, 640), (640, 1))
    assert_size_stride(permute_371, (640, 128), (128, 1))
    assert_size_stride(unsqueeze_193, (1, 128), (128, 1))
    assert_size_stride(permute_375, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_197, (1, 256), (256, 1))
    assert_size_stride(permute_379, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_201, (1, 128), (128, 1))
    assert_size_stride(permute_383, (128, 128), (128, 1))
    assert_size_stride(permute_386, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_387, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_24, (8, 4, 196, 196), (153664, 1, 784, 4))
    assert_size_stride(permute_388, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_389, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_205, (1, 256), (256, 1))
    assert_size_stride(permute_395, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_209, (1, 128), (128, 1))
    assert_size_stride(permute_399, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_213, (1, 256), (256, 1))
    assert_size_stride(permute_403, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_217, (1, 128), (128, 1))
    assert_size_stride(permute_407, (128, 128), (128, 1))
    assert_size_stride(permute_410, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_411, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_25, (8, 4, 196, 196), (153664, 1, 784, 4))
    assert_size_stride(permute_412, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_413, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_221, (1, 256), (256, 1))
    assert_size_stride(permute_419, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_225, (1, 128), (128, 1))
    assert_size_stride(permute_423, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_229, (1, 256), (256, 1))
    assert_size_stride(permute_427, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_233, (1, 128), (128, 1))
    assert_size_stride(permute_431, (128, 128), (128, 1))
    assert_size_stride(permute_434, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_435, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_26, (8, 4, 196, 196), (153664, 1, 784, 4))
    assert_size_stride(permute_436, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_437, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_237, (1, 256), (256, 1))
    assert_size_stride(permute_443, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_241, (1, 128), (128, 1))
    assert_size_stride(permute_447, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_245, (1, 256), (256, 1))
    assert_size_stride(permute_451, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_249, (1, 128), (128, 1))
    assert_size_stride(permute_455, (128, 128), (128, 1))
    assert_size_stride(permute_458, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_459, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_27, (8, 4, 196, 196), (153664, 1, 784, 4))
    assert_size_stride(permute_460, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_461, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_253, (1, 256), (256, 1))
    assert_size_stride(permute_467, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_259, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_271, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_283, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_295, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf3 = empty((8, 1000), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_div_0(c_void_p(mean.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del tangents_1
    buf4 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, permute_117, out=buf4)
    del permute_117
    buf5 = empty((1000, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (1000, 8), (1, 1000), 0), clone_82, out=buf5)
    del clone_82
    buf6 = empty((1, 1000), device='cpu', dtype=torch.float32)
    cpp_fused_sum_1(c_void_p(buf3.data_ptr()), c_void_p(buf6.data_ptr()))
    buf10 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, permute_121, out=buf10)
    del permute_121
    buf12 = empty((384, ), device='cpu', dtype=torch.float32)
    buf7 = empty((384, ), device='cpu', dtype=torch.float32)
    buf8 = empty((384, ), device='cpu', dtype=torch.float32)
    buf13 = empty((384, ), device='cpu', dtype=torch.float32)
    buf9 = empty((384, ), device='cpu', dtype=torch.float32)
    buf14 = empty((384, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_2(c_void_p(buf10.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(mean.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf14.data_ptr()))
    buf11 = empty((1000, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (1000, 8), (1, 1000), 0), clone_81, out=buf11)
    del buf3
    del clone_81
    buf15 = buf10; del buf10  # reuse
    buf16 = empty((384, ), device='cpu', dtype=torch.float32)
    buf17 = empty((384, ), device='cpu', dtype=torch.float32)
    buf18 = empty((384, ), device='cpu', dtype=torch.float32)
    buf19 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_batch_norm_backward_3(c_void_p(buf15.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(mean.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(mm_57.data_ptr()), c_void_p(unsqueeze_25.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del mean
    del mm_57
    del primals_199
    del primals_201
    del primals_205
    del squeeze_184
    del unsqueeze_25
    buf20 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (384, 128), (1, 384), 0), view_347, out=buf20)
    del view_347
    buf21 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (128, 384), (384, 1), 0), permute_127, out=buf21)
    del permute_127
    buf22 = empty((768, ), device='cpu', dtype=torch.float32)
    buf23 = empty((768, ), device='cpu', dtype=torch.float32)
    buf24 = empty((768, ), device='cpu', dtype=torch.float32)
    buf25 = buf21; del buf21  # reuse
    cpp_fused_native_batch_norm_backward_4(c_void_p(buf25.data_ptr()), c_void_p(view_346.data_ptr()), c_void_p(mm_56.data_ptr()), c_void_p(unsqueeze_29.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()))
    del mm_56
    del primals_196
    del squeeze_181
    del unsqueeze_29
    del view_346
    buf26 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf25, (768, 128), (1, 768), 0), view_343, out=buf26)
    del view_343
    buf27 = buf19; del buf19  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf25, (128, 768), (768, 1), 0), permute_131, out=buf27)
    del permute_131
    buf28 = buf8; del buf8  # reuse
    buf29 = buf17; del buf17  # reuse
    buf30 = buf13; del buf13  # reuse
    buf31 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_5(c_void_p(buf15.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(mm_55.data_ptr()), c_void_p(unsqueeze_33.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del mm_55
    del primals_193
    del squeeze_178
    del unsqueeze_33
    buf32 = empty((384, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (384, 128), (1, 384), 0), view_339, out=buf32)
    del view_339
    buf33 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (128, 384), (384, 1), 0), permute_135, out=buf33)
    del permute_135
    buf34 = reinterpret_tensor(buf31, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf31  # reuse
    cpp_fused_clone_6(c_void_p(view_338.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del view_338
    buf35 = reinterpret_tensor(buf33, (96, 16, 32), (512, 32, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_138, reinterpret_tensor(buf34, (96, 16, 32), (512, 32, 1), 0), out=buf35)
    del permute_138
    buf36 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf34, (96, 16, 32), (512, 32, 1), 0), permute_139, out=buf36)
    del permute_139
    buf37 = empty_strided((8, 12, 16, 1), (192, 16, 1, 1536), device='cpu', dtype=torch.float32)
    buf38 = reinterpret_tensor(buf4, (1, 12, 16, 16), (3072, 256, 16, 1), 0); del buf4  # reuse
    buf39 = empty((12, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_7(c_void_p(buf36.data_ptr()), c_void_p(alias_14.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    aten.index_put_(buf39, [None, primals_222], reinterpret_tensor(buf38, (12, 16, 16), (256, 16, 1), 0), True)
    del primals_222
    buf42 = reinterpret_tensor(buf36, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf36  # reuse
    cpp_fused__softmax_backward_data_mul_8(c_void_p(buf42.data_ptr()), c_void_p(alias_14.data_ptr()), c_void_p(buf37.data_ptr()))
    del alias_14
    buf43 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_140, reinterpret_tensor(buf42, (96, 16, 16), (256, 16, 1), 0), out=buf43)
    del permute_140
    buf44 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf42, (96, 16, 16), (256, 16, 1), 0), permute_141, out=buf44)
    del permute_141
    buf45 = buf23; del buf23  # reuse
    buf46 = empty((768, ), device='cpu', dtype=torch.float32)
    buf47 = buf25; del buf25  # reuse
    buf48 = buf46; del buf46  # reuse
    cpp_fused_native_batch_norm_backward_9(c_void_p(buf48.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(mm_54.data_ptr()), c_void_p(unsqueeze_37.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()))
    del mm_54
    del primals_190
    del squeeze_175
    del unsqueeze_37
    buf49 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (768, 128), (1, 768), 0), view_327, out=buf49)
    del view_327
    buf50 = reinterpret_tensor(buf35, (128, 384), (384, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (128, 768), (768, 1), 0), permute_147, out=buf50)
    del permute_147
    buf51 = buf29; del buf29  # reuse
    buf52 = reinterpret_tensor(buf1, (384, ), (1, ), 0); del buf1  # reuse
    buf53 = reinterpret_tensor(buf34, (128, 384), (384, 1), 0); del buf34  # reuse
    buf54 = buf52; del buf52  # reuse
    cpp_fused_native_batch_norm_backward_10(c_void_p(buf54.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(mm_53.data_ptr()), c_void_p(unsqueeze_41.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    del mm_53
    del primals_187
    del squeeze_172
    del unsqueeze_41
    buf55 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf53, (384, 128), (1, 384), 0), view_323, out=buf55)
    del view_323
    buf56 = buf47; del buf47  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf53, (128, 384), (384, 1), 0), permute_151, out=buf56)
    del permute_151
    buf57 = empty((768, ), device='cpu', dtype=torch.float32)
    buf58 = empty((768, ), device='cpu', dtype=torch.float32)
    buf59 = empty((768, ), device='cpu', dtype=torch.float32)
    buf60 = buf56; del buf56  # reuse
    cpp_fused_native_batch_norm_backward_11(c_void_p(buf60.data_ptr()), c_void_p(view_322.data_ptr()), c_void_p(mm_52.data_ptr()), c_void_p(unsqueeze_45.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()))
    del mm_52
    del primals_184
    del squeeze_169
    del unsqueeze_45
    del view_322
    buf61 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (768, 128), (1, 768), 0), view_319, out=buf61)
    del view_319
    buf62 = buf53; del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (128, 768), (768, 1), 0), permute_155, out=buf62)
    del permute_155
    buf63 = reinterpret_tensor(buf0, (384, ), (1, ), 0); del buf0  # reuse
    buf64 = empty((384, ), device='cpu', dtype=torch.float32)
    buf65 = empty((128, 384), device='cpu', dtype=torch.float32)
    buf67 = buf65; del buf65  # reuse
    buf66 = buf64; del buf64  # reuse
    cpp_fused_native_batch_norm_backward_12(c_void_p(buf67.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(mm_51.data_ptr()), c_void_p(unsqueeze_49.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf63.data_ptr()))
    del mm_51
    del primals_181
    del squeeze_166
    del unsqueeze_49
    buf68 = empty((384, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (384, 128), (1, 384), 0), view_315, out=buf68)
    del view_315
    buf69 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (128, 384), (384, 1), 0), permute_159, out=buf69)
    del permute_159
    buf70 = reinterpret_tensor(buf67, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf67  # reuse
    cpp_fused_clone_13(c_void_p(view_314.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del view_314
    buf71 = reinterpret_tensor(buf69, (96, 16, 32), (512, 32, 1), 0); del buf69  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_162, reinterpret_tensor(buf70, (96, 16, 32), (512, 32, 1), 0), out=buf71)
    del permute_162
    buf72 = buf44; del buf44  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf70, (96, 16, 32), (512, 32, 1), 0), permute_163, out=buf72)
    del permute_163
    buf73 = buf37; del buf37  # reuse
    buf74 = buf38; del buf38  # reuse
    buf75 = empty((12, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_14(c_void_p(buf72.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    aten.index_put_(buf75, [None, primals_221], reinterpret_tensor(buf74, (12, 16, 16), (256, 16, 1), 0), True)
    del buf74
    del primals_221
    buf78 = reinterpret_tensor(buf72, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf72  # reuse
    cpp_fused__softmax_backward_data_mul_15(c_void_p(buf78.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(buf73.data_ptr()))
    del alias_15
    buf79 = buf43; del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_164, reinterpret_tensor(buf78, (96, 16, 16), (256, 16, 1), 0), out=buf79)
    del permute_164
    buf80 = reinterpret_tensor(buf42, (96, 16, 16), (256, 16, 1), 0); del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf78, (96, 16, 16), (256, 16, 1), 0), permute_165, out=buf80)
    del permute_165
    buf81 = buf58; del buf58  # reuse
    buf82 = empty((768, ), device='cpu', dtype=torch.float32)
    buf83 = buf60; del buf60  # reuse
    buf84 = buf82; del buf82  # reuse
    cpp_fused_native_batch_norm_backward_16(c_void_p(buf84.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(mm_50.data_ptr()), c_void_p(unsqueeze_53.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()))
    del mm_50
    del primals_178
    del squeeze_163
    del unsqueeze_53
    buf85 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (768, 128), (1, 768), 0), view_303, out=buf85)
    del view_303
    buf86 = reinterpret_tensor(buf71, (128, 384), (384, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (128, 768), (768, 1), 0), permute_171, out=buf86)
    del permute_171
    buf87 = reinterpret_tensor(buf27, (8, 16, 384), (6144, 384, 1), 0); del buf27  # reuse
    buf88 = empty((384, ), device='cpu', dtype=torch.float32)
    buf89 = empty((384, ), device='cpu', dtype=torch.float32)
    buf90 = empty((384, ), device='cpu', dtype=torch.float32)
    buf91 = reinterpret_tensor(buf70, (128, 384), (384, 1), 0); del buf70  # reuse
    cpp_fused_add_div_native_batch_norm_backward_17(c_void_p(buf87.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(mm_49.data_ptr()), c_void_p(unsqueeze_57.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    del mm_49
    del primals_175
    del squeeze_160
    del unsqueeze_57
    buf92 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (384, 128), (1, 384), 0), view_299, out=buf92)
    del view_299
    buf93 = buf83; del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (128, 384), (384, 1), 0), permute_175, out=buf93)
    del permute_175
    buf94 = empty((768, ), device='cpu', dtype=torch.float32)
    buf95 = empty((768, ), device='cpu', dtype=torch.float32)
    buf96 = empty((768, ), device='cpu', dtype=torch.float32)
    buf97 = buf93; del buf93  # reuse
    cpp_fused_native_batch_norm_backward_18(c_void_p(buf97.data_ptr()), c_void_p(view_298.data_ptr()), c_void_p(mm_48.data_ptr()), c_void_p(unsqueeze_61.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del mm_48
    del primals_172
    del squeeze_157
    del unsqueeze_61
    del view_298
    buf98 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (768, 128), (1, 768), 0), view_295, out=buf98)
    del view_295
    buf99 = buf91; del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (128, 768), (768, 1), 0), permute_179, out=buf99)
    del permute_179
    buf100 = buf89; del buf89  # reuse
    buf101 = empty((384, ), device='cpu', dtype=torch.float32)
    buf102 = empty((384, ), device='cpu', dtype=torch.float32)
    buf103 = buf86; del buf86  # reuse
    cpp_fused_native_batch_norm_backward_19(c_void_p(buf87.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(mm_47.data_ptr()), c_void_p(unsqueeze_65.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del mm_47
    del primals_169
    del squeeze_154
    del unsqueeze_65
    buf104 = empty((384, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (384, 128), (1, 384), 0), view_291, out=buf104)
    del view_291
    buf105 = buf62; del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (128, 384), (384, 1), 0), permute_183, out=buf105)
    del permute_183
    buf106 = reinterpret_tensor(buf103, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf103  # reuse
    cpp_fused_clone_20(c_void_p(view_290.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del view_290
    buf107 = reinterpret_tensor(buf105, (96, 16, 32), (512, 32, 1), 0); del buf105  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_186, reinterpret_tensor(buf106, (96, 16, 32), (512, 32, 1), 0), out=buf107)
    del permute_186
    buf108 = buf80; del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf106, (96, 16, 32), (512, 32, 1), 0), permute_187, out=buf108)
    del permute_187
    buf109 = buf73; del buf73  # reuse
    buf110 = reinterpret_tensor(buf15, (1, 12, 16, 16), (3072, 256, 16, 1), 0); del buf15  # reuse
    buf111 = empty((12, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_21(c_void_p(buf108.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    aten.index_put_(buf111, [None, primals_220], reinterpret_tensor(buf110, (12, 16, 16), (256, 16, 1), 0), True)
    del primals_220
    buf114 = reinterpret_tensor(buf108, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf108  # reuse
    cpp_fused__softmax_backward_data_mul_22(c_void_p(buf114.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(buf109.data_ptr()))
    del alias_16
    buf115 = buf79; del buf79  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_188, reinterpret_tensor(buf114, (96, 16, 16), (256, 16, 1), 0), out=buf115)
    del permute_188
    buf116 = reinterpret_tensor(buf78, (96, 16, 16), (256, 16, 1), 0); del buf78  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf114, (96, 16, 16), (256, 16, 1), 0), permute_189, out=buf116)
    del permute_189
    buf117 = buf95; del buf95  # reuse
    buf118 = empty((768, ), device='cpu', dtype=torch.float32)
    buf119 = buf97; del buf97  # reuse
    buf120 = buf118; del buf118  # reuse
    cpp_fused_native_batch_norm_backward_23(c_void_p(buf120.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(mm_46.data_ptr()), c_void_p(unsqueeze_69.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    del mm_46
    del primals_166
    del squeeze_151
    del unsqueeze_69
    buf121 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (768, 128), (1, 768), 0), view_279, out=buf121)
    del view_279
    buf122 = reinterpret_tensor(buf107, (128, 384), (384, 1), 0); del buf107  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (128, 768), (768, 1), 0), permute_195, out=buf122)
    del permute_195
    buf123 = buf101; del buf101  # reuse
    buf124 = empty((384, ), device='cpu', dtype=torch.float32)
    buf125 = reinterpret_tensor(buf106, (128, 384), (384, 1), 0); del buf106  # reuse
    buf126 = buf124; del buf124  # reuse
    cpp_fused_native_batch_norm_backward_24(c_void_p(buf126.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(mm_45.data_ptr()), c_void_p(unsqueeze_73.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()))
    del mm_45
    del primals_163
    del squeeze_148
    del unsqueeze_73
    buf127 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (384, 128), (1, 384), 0), view_275, out=buf127)
    del view_275
    buf128 = buf119; del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (128, 384), (384, 1), 0), permute_199, out=buf128)
    del permute_199
    buf129 = empty((768, ), device='cpu', dtype=torch.float32)
    buf130 = empty((768, ), device='cpu', dtype=torch.float32)
    buf131 = empty((768, ), device='cpu', dtype=torch.float32)
    buf132 = buf128; del buf128  # reuse
    cpp_fused_native_batch_norm_backward_25(c_void_p(buf132.data_ptr()), c_void_p(view_274.data_ptr()), c_void_p(mm_44.data_ptr()), c_void_p(unsqueeze_77.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del mm_44
    del primals_160
    del squeeze_145
    del unsqueeze_77
    del view_274
    buf133 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (768, 128), (1, 768), 0), view_271, out=buf133)
    del view_271
    buf134 = buf125; del buf125  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (128, 768), (768, 1), 0), permute_203, out=buf134)
    del permute_203
    buf135 = empty((384, ), device='cpu', dtype=torch.float32)
    buf136 = empty((384, ), device='cpu', dtype=torch.float32)
    buf137 = buf50; del buf50  # reuse
    buf139 = buf137; del buf137  # reuse
    buf138 = buf136; del buf136  # reuse
    cpp_fused_native_batch_norm_backward_26(c_void_p(buf139.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(mm_43.data_ptr()), c_void_p(unsqueeze_81.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf135.data_ptr()))
    del mm_43
    del primals_157
    del squeeze_142
    del unsqueeze_81
    buf140 = empty((384, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (384, 128), (1, 384), 0), view_267, out=buf140)
    del view_267
    buf141 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (128, 384), (384, 1), 0), permute_207, out=buf141)
    del permute_207
    buf142 = reinterpret_tensor(buf139, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf139  # reuse
    cpp_fused_clone_27(c_void_p(view_266.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    del view_266
    buf143 = reinterpret_tensor(buf141, (96, 16, 32), (512, 32, 1), 0); del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_210, reinterpret_tensor(buf142, (96, 16, 32), (512, 32, 1), 0), out=buf143)
    del permute_210
    buf144 = buf116; del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf142, (96, 16, 32), (512, 32, 1), 0), permute_211, out=buf144)
    del permute_211
    buf145 = buf109; del buf109  # reuse
    buf146 = buf110; del buf110  # reuse
    buf147 = empty((12, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_new_zeros_sum_28(c_void_p(buf144.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    aten.index_put_(buf147, [None, primals_219], reinterpret_tensor(buf146, (12, 16, 16), (256, 16, 1), 0), True)
    del buf146
    del primals_219
    buf150 = reinterpret_tensor(buf144, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf144  # reuse
    cpp_fused__softmax_backward_data_mul_29(c_void_p(buf150.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(buf145.data_ptr()))
    del alias_17
    del buf145
    buf151 = buf115; del buf115  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_212, reinterpret_tensor(buf150, (96, 16, 16), (256, 16, 1), 0), out=buf151)
    del permute_212
    buf152 = reinterpret_tensor(buf114, (96, 16, 16), (256, 16, 1), 0); del buf114  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf150, (96, 16, 16), (256, 16, 1), 0), permute_213, out=buf152)
    del buf150
    del permute_213
    buf153 = buf130; del buf130  # reuse
    buf154 = empty((768, ), device='cpu', dtype=torch.float32)
    buf155 = buf132; del buf132  # reuse
    buf156 = buf154; del buf154  # reuse
    cpp_fused_native_batch_norm_backward_30(c_void_p(buf156.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(mm_42.data_ptr()), c_void_p(unsqueeze_85.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()))
    del buf151
    del buf152
    del mm_42
    del primals_154
    del squeeze_139
    del unsqueeze_85
    buf157 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (768, 128), (1, 768), 0), view_255, out=buf157)
    del view_255
    buf158 = reinterpret_tensor(buf143, (128, 384), (384, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (128, 768), (768, 1), 0), permute_219, out=buf158)
    del permute_219
    buf159 = reinterpret_tensor(buf122, (8, 16, 384), (6144, 384, 1), 0); del buf122  # reuse
    buf160 = empty((384, ), device='cpu', dtype=torch.float32)
    buf161 = empty((384, ), device='cpu', dtype=torch.float32)
    buf162 = empty((384, ), device='cpu', dtype=torch.float32)
    buf163 = reinterpret_tensor(buf142, (128, 384), (384, 1), 0); del buf142  # reuse
    cpp_fused_add_native_batch_norm_backward_31(c_void_p(buf159.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(mm_41.data_ptr()), c_void_p(unsqueeze_89.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del buf134
    del buf158
    del buf87
    del buf99
    del mm_41
    del primals_151
    del squeeze_136
    del unsqueeze_89
    buf164 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (384, 128), (1, 384), 0), view_251, out=buf164)
    del view_251
    buf165 = buf155; del buf155  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (128, 384), (384, 1), 0), permute_223, out=buf165)
    del permute_223
    buf166 = empty((768, ), device='cpu', dtype=torch.float32)
    buf167 = empty((768, ), device='cpu', dtype=torch.float32)
    buf168 = empty((768, ), device='cpu', dtype=torch.float32)
    buf169 = buf165; del buf165  # reuse
    cpp_fused_native_batch_norm_backward_32(c_void_p(buf169.data_ptr()), c_void_p(view_250.data_ptr()), c_void_p(mm_40.data_ptr()), c_void_p(unsqueeze_93.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    del buf167
    del mm_40
    del primals_148
    del squeeze_133
    del unsqueeze_93
    del view_250
    buf170 = empty((768, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (768, 128), (1, 768), 0), view_247, out=buf170)
    del view_247
    buf171 = buf163; del buf163  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (128, 768), (768, 1), 0), permute_227, out=buf171)
    del buf169
    del permute_227
    buf172 = buf161; del buf161  # reuse
    buf173 = empty((384, ), device='cpu', dtype=torch.float32)
    buf174 = empty((384, ), device='cpu', dtype=torch.float32)
    buf175 = reinterpret_tensor(buf159, (128, 384), (384, 1), 0); del buf159  # reuse
    cpp_fused_native_batch_norm_backward_33(c_void_p(buf175.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(mm_39.data_ptr()), c_void_p(unsqueeze_97.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del buf171
    del buf173
    del mm_39
    del primals_145
    del squeeze_130
    del unsqueeze_97
    buf176 = empty((384, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (384, 128), (1, 384), 0), view_243, out=buf176)
    del view_243
    buf177 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (128, 384), (384, 1), 0), permute_231, out=buf177)
    del buf175
    del permute_231
    buf178 = empty((8, 16, 16, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_34(c_void_p(view_242.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    del view_242
    buf179 = empty((128, 49, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_234, reinterpret_tensor(buf178, (128, 16, 64), (1024, 64, 1), 0), out=buf179)
    del permute_234
    buf180 = empty((128, 16, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf178, (128, 16, 64), (1024, 64, 1), 0), permute_235, out=buf180)
    del permute_235
    buf181 = empty_strided((8, 16, 16, 1), (256, 16, 1, 2048), device='cpu', dtype=torch.float32)
    buf182 = empty((1, 16, 16, 49), device='cpu', dtype=torch.float32)
    buf183 = empty((16, 49), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_new_zeros_sum_35(c_void_p(buf180.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    aten.index_put_(buf183, [None, primals_218], reinterpret_tensor(buf182, (16, 16, 49), (784, 49, 1), 0), True)
    del buf182
    del primals_218
    buf186 = reinterpret_tensor(buf180, (8, 16, 16, 49), (12544, 784, 49, 1), 0); del buf180  # reuse
    cpp_fused__softmax_backward_data_mul_36(c_void_p(buf186.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(buf181.data_ptr()))
    del alias_18
    del buf181
    buf187 = empty((128, 16, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_236, reinterpret_tensor(buf186, (128, 16, 49), (784, 49, 1), 0), out=buf187)
    del permute_236
    buf188 = empty((128, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf186, (128, 16, 49), (784, 49, 1), 0), permute_237, out=buf188)
    del permute_237
    buf189 = empty((256, ), device='cpu', dtype=torch.float32)
    buf190 = empty((256, ), device='cpu', dtype=torch.float32)
    buf191 = empty((256, ), device='cpu', dtype=torch.float32)
    buf192 = empty((128, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_37(c_void_p(buf188.data_ptr()), c_void_p(mm_38.data_ptr()), c_void_p(unsqueeze_101.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del mm_38
    del primals_142
    del squeeze_127
    del unsqueeze_101
    buf193 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (256, 128), (1, 256), 0), view_231, out=buf193)
    del view_231
    buf194 = reinterpret_tensor(buf188, (128, 256), (256, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (128, 256), (256, 1), 0), permute_241, out=buf194)
    del permute_241
    buf195 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf196 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf197 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf198 = empty((392, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_38(c_void_p(buf187.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(mm_37.data_ptr()), c_void_p(unsqueeze_105.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    del buf196
    del mm_37
    del primals_139
    del squeeze_124
    del unsqueeze_105
    buf199 = empty((1280, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (1280, 392), (1, 1280), 0), view_224, out=buf199)
    del view_224
    buf200 = reinterpret_tensor(buf187, (392, 256), (256, 1), 0); del buf187  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (392, 1280), (1280, 1), 0), permute_247, out=buf200)
    del buf198
    del permute_247
    buf201 = buf190; del buf190  # reuse
    buf202 = empty((256, ), device='cpu', dtype=torch.float32)
    buf203 = empty((256, ), device='cpu', dtype=torch.float32)
    buf204 = reinterpret_tensor(buf186, (392, 256), (256, 1), 0); del buf186  # reuse
    cpp_fused_native_batch_norm_backward_39(c_void_p(buf194.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(mm_36.data_ptr()), c_void_p(unsqueeze_109.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del mm_36
    del primals_136
    del squeeze_121
    del unsqueeze_109
    buf205 = reinterpret_tensor(buf178, (256, 512), (512, 1), 0); del buf178  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (256, 392), (1, 256), 0), view_220, out=buf205)
    del view_220
    buf206 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (392, 256), (256, 1), 0), permute_251, out=buf206)
    del permute_251
    buf207 = empty((512, ), device='cpu', dtype=torch.float32)
    buf208 = empty((512, ), device='cpu', dtype=torch.float32)
    buf209 = empty((512, ), device='cpu', dtype=torch.float32)
    buf210 = buf206; del buf206  # reuse
    cpp_fused_native_batch_norm_backward_40(c_void_p(buf210.data_ptr()), c_void_p(view_219.data_ptr()), c_void_p(mm_35.data_ptr()), c_void_p(unsqueeze_113.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del mm_35
    del primals_133
    del squeeze_118
    del unsqueeze_113
    del view_219
    buf211 = reinterpret_tensor(buf177, (512, 256), (256, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (512, 392), (1, 512), 0), view_216, out=buf211)
    del view_216
    buf212 = buf204; del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (392, 512), (512, 1), 0), permute_255, out=buf212)
    del permute_255
    buf213 = buf202; del buf202  # reuse
    buf214 = empty((256, ), device='cpu', dtype=torch.float32)
    buf215 = empty((392, 256), device='cpu', dtype=torch.float32)
    buf216 = buf214; del buf214  # reuse
    cpp_fused_native_batch_norm_backward_41(c_void_p(buf216.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(mm_34.data_ptr()), c_void_p(unsqueeze_117.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()))
    del mm_34
    del primals_130
    del squeeze_115
    del unsqueeze_117
    buf217 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (256, 392), (1, 256), 0), view_212, out=buf217)
    del view_212
    buf218 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (392, 256), (256, 1), 0), permute_259, out=buf218)
    del permute_259
    buf219 = reinterpret_tensor(buf215, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf215  # reuse
    cpp_fused_clone_42(c_void_p(view_211.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    del view_211
    buf220 = reinterpret_tensor(buf218, (64, 49, 32), (1568, 32, 1), 0); del buf218  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_262, reinterpret_tensor(buf219, (64, 49, 32), (1568, 32, 1), 0), out=buf220)
    del permute_262
    buf221 = empty((64, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf219, (64, 49, 32), (1568, 32, 1), 0), permute_263, out=buf221)
    del permute_263
    buf222 = empty_strided((8, 8, 49, 1), (392, 49, 1, 3136), device='cpu', dtype=torch.float32)
    buf223 = empty((1, 8, 49, 49), device='cpu', dtype=torch.float32)
    buf224 = empty((8, 49), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_43(c_void_p(buf221.data_ptr()), c_void_p(alias_19.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    aten.index_put_(buf224, [None, primals_217], reinterpret_tensor(buf223, (8, 49, 49), (2401, 49, 1), 0), True)
    del primals_217
    buf227 = reinterpret_tensor(buf221, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf221  # reuse
    cpp_fused__softmax_backward_data_mul_44(c_void_p(buf227.data_ptr()), c_void_p(alias_19.data_ptr()), c_void_p(buf222.data_ptr()))
    del alias_19
    buf228 = empty((64, 16, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_264, reinterpret_tensor(buf227, (64, 49, 49), (2401, 49, 1), 0), out=buf228)
    del permute_264
    buf229 = empty((64, 49, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (64, 49, 49), (2401, 49, 1), 0), permute_265, out=buf229)
    del permute_265
    buf230 = buf208; del buf208  # reuse
    buf231 = empty((512, ), device='cpu', dtype=torch.float32)
    buf232 = buf210; del buf210  # reuse
    buf233 = buf231; del buf231  # reuse
    cpp_fused_native_batch_norm_backward_45(c_void_p(buf233.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(mm_33.data_ptr()), c_void_p(unsqueeze_121.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    del mm_33
    del primals_127
    del squeeze_112
    del unsqueeze_121
    buf234 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (512, 392), (1, 512), 0), view_200, out=buf234)
    del view_200
    buf235 = reinterpret_tensor(buf220, (392, 256), (256, 1), 0); del buf220  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (392, 512), (512, 1), 0), permute_271, out=buf235)
    del permute_271
    buf236 = empty((256, ), device='cpu', dtype=torch.float32)
    buf237 = empty((256, ), device='cpu', dtype=torch.float32)
    buf238 = reinterpret_tensor(buf219, (392, 256), (256, 1), 0); del buf219  # reuse
    buf240 = buf238; del buf238  # reuse
    buf239 = buf237; del buf237  # reuse
    cpp_fused_native_batch_norm_backward_46(c_void_p(buf240.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(mm_32.data_ptr()), c_void_p(unsqueeze_125.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf236.data_ptr()))
    del mm_32
    del primals_124
    del squeeze_109
    del unsqueeze_125
    buf241 = empty((256, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (256, 392), (1, 256), 0), view_196, out=buf241)
    del view_196
    buf242 = buf232; del buf232  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (392, 256), (256, 1), 0), permute_275, out=buf242)
    del permute_275
    buf243 = empty((512, ), device='cpu', dtype=torch.float32)
    buf244 = empty((512, ), device='cpu', dtype=torch.float32)
    buf245 = empty((512, ), device='cpu', dtype=torch.float32)
    buf246 = buf242; del buf242  # reuse
    cpp_fused_native_batch_norm_backward_47(c_void_p(buf246.data_ptr()), c_void_p(view_195.data_ptr()), c_void_p(mm_31.data_ptr()), c_void_p(unsqueeze_129.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    del mm_31
    del primals_121
    del squeeze_106
    del unsqueeze_129
    del view_195
    buf247 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf246, (512, 392), (1, 512), 0), view_192, out=buf247)
    del view_192
    buf248 = buf240; del buf240  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf246, (392, 512), (512, 1), 0), permute_279, out=buf248)
    del permute_279
    buf249 = reinterpret_tensor(buf200, (8, 49, 256), (12544, 256, 1), 0); del buf200  # reuse
    buf250 = empty((256, ), device='cpu', dtype=torch.float32)
    buf251 = empty((256, ), device='cpu', dtype=torch.float32)
    buf252 = empty((256, ), device='cpu', dtype=torch.float32)
    buf253 = empty((392, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_batch_norm_backward_48(c_void_p(buf249.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(mm_30.data_ptr()), c_void_p(unsqueeze_133.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del mm_30
    del primals_118
    del squeeze_103
    del unsqueeze_133
    buf254 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf253, (256, 392), (1, 256), 0), view_188, out=buf254)
    del view_188
    buf255 = buf248; del buf248  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf253, (392, 256), (256, 1), 0), permute_283, out=buf255)
    del permute_283
    buf256 = reinterpret_tensor(buf253, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf253  # reuse
    cpp_fused_clone_49(c_void_p(view_187.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del view_187
    buf257 = reinterpret_tensor(buf255, (64, 49, 32), (1568, 32, 1), 0); del buf255  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_286, reinterpret_tensor(buf256, (64, 49, 32), (1568, 32, 1), 0), out=buf257)
    del permute_286
    buf258 = reinterpret_tensor(buf227, (64, 49, 49), (2401, 49, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf256, (64, 49, 32), (1568, 32, 1), 0), permute_287, out=buf258)
    del permute_287
    buf259 = buf222; del buf222  # reuse
    buf260 = buf223; del buf223  # reuse
    buf261 = empty((8, 49), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_50(c_void_p(buf258.data_ptr()), c_void_p(alias_20.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    aten.index_put_(buf261, [None, primals_216], reinterpret_tensor(buf260, (8, 49, 49), (2401, 49, 1), 0), True)
    del primals_216
    buf264 = reinterpret_tensor(buf258, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf258  # reuse
    cpp_fused__softmax_backward_data_mul_51(c_void_p(buf264.data_ptr()), c_void_p(alias_20.data_ptr()), c_void_p(buf259.data_ptr()))
    del alias_20
    buf265 = reinterpret_tensor(buf229, (64, 16, 49), (784, 49, 1), 0); del buf229  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_288, reinterpret_tensor(buf264, (64, 49, 49), (2401, 49, 1), 0), out=buf265)
    del permute_288
    buf266 = reinterpret_tensor(buf228, (64, 49, 16), (784, 16, 1), 0); del buf228  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf264, (64, 49, 49), (2401, 49, 1), 0), permute_289, out=buf266)
    del permute_289
    buf267 = buf244; del buf244  # reuse
    buf268 = empty((512, ), device='cpu', dtype=torch.float32)
    buf269 = buf246; del buf246  # reuse
    buf270 = buf268; del buf268  # reuse
    cpp_fused_native_batch_norm_backward_52(c_void_p(buf270.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(mm_29.data_ptr()), c_void_p(unsqueeze_137.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()))
    del mm_29
    del primals_115
    del squeeze_100
    del unsqueeze_137
    buf271 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (512, 392), (1, 512), 0), view_176, out=buf271)
    del view_176
    buf272 = reinterpret_tensor(buf257, (392, 256), (256, 1), 0); del buf257  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (392, 512), (512, 1), 0), permute_295, out=buf272)
    del permute_295
    buf273 = buf251; del buf251  # reuse
    buf274 = empty((256, ), device='cpu', dtype=torch.float32)
    buf275 = empty((256, ), device='cpu', dtype=torch.float32)
    buf276 = reinterpret_tensor(buf256, (392, 256), (256, 1), 0); del buf256  # reuse
    cpp_fused_native_batch_norm_backward_53(c_void_p(buf249.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(mm_28.data_ptr()), c_void_p(unsqueeze_141.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del mm_28
    del primals_112
    del squeeze_97
    del unsqueeze_141
    buf277 = empty((256, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (256, 392), (1, 256), 0), view_172, out=buf277)
    del view_172
    buf278 = buf269; del buf269  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (392, 256), (256, 1), 0), permute_299, out=buf278)
    del permute_299
    buf279 = empty((512, ), device='cpu', dtype=torch.float32)
    buf280 = empty((512, ), device='cpu', dtype=torch.float32)
    buf281 = empty((512, ), device='cpu', dtype=torch.float32)
    buf282 = buf278; del buf278  # reuse
    cpp_fused_native_batch_norm_backward_54(c_void_p(buf282.data_ptr()), c_void_p(view_171.data_ptr()), c_void_p(mm_27.data_ptr()), c_void_p(unsqueeze_145.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del mm_27
    del primals_109
    del squeeze_94
    del unsqueeze_145
    del view_171
    buf283 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (512, 392), (1, 512), 0), view_168, out=buf283)
    del view_168
    buf284 = buf276; del buf276  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (392, 512), (512, 1), 0), permute_303, out=buf284)
    del permute_303
    buf285 = buf274; del buf274  # reuse
    buf286 = empty((256, ), device='cpu', dtype=torch.float32)
    buf287 = buf235; del buf235  # reuse
    buf288 = buf286; del buf286  # reuse
    cpp_fused_native_batch_norm_backward_55(c_void_p(buf288.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(mm_26.data_ptr()), c_void_p(unsqueeze_149.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()))
    del mm_26
    del primals_106
    del squeeze_91
    del unsqueeze_149
    buf289 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (256, 392), (1, 256), 0), view_164, out=buf289)
    del view_164
    buf290 = buf212; del buf212  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (392, 256), (256, 1), 0), permute_307, out=buf290)
    del permute_307
    buf291 = reinterpret_tensor(buf287, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf287  # reuse
    cpp_fused_clone_56(c_void_p(view_163.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del view_163
    buf292 = reinterpret_tensor(buf290, (64, 49, 32), (1568, 32, 1), 0); del buf290  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_310, reinterpret_tensor(buf291, (64, 49, 32), (1568, 32, 1), 0), out=buf292)
    del permute_310
    buf293 = reinterpret_tensor(buf264, (64, 49, 49), (2401, 49, 1), 0); del buf264  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf291, (64, 49, 32), (1568, 32, 1), 0), permute_311, out=buf293)
    del permute_311
    buf294 = buf259; del buf259  # reuse
    buf295 = buf260; del buf260  # reuse
    buf296 = empty((8, 49), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_57(c_void_p(buf293.data_ptr()), c_void_p(alias_21.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    aten.index_put_(buf296, [None, primals_215], reinterpret_tensor(buf295, (8, 49, 49), (2401, 49, 1), 0), True)
    del primals_215
    buf299 = reinterpret_tensor(buf293, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf293  # reuse
    cpp_fused__softmax_backward_data_mul_58(c_void_p(buf299.data_ptr()), c_void_p(alias_21.data_ptr()), c_void_p(buf294.data_ptr()))
    del alias_21
    buf300 = reinterpret_tensor(buf266, (64, 16, 49), (784, 49, 1), 0); del buf266  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_312, reinterpret_tensor(buf299, (64, 49, 49), (2401, 49, 1), 0), out=buf300)
    del permute_312
    buf301 = reinterpret_tensor(buf265, (64, 49, 16), (784, 16, 1), 0); del buf265  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf299, (64, 49, 49), (2401, 49, 1), 0), permute_313, out=buf301)
    del permute_313
    buf302 = buf280; del buf280  # reuse
    buf303 = empty((512, ), device='cpu', dtype=torch.float32)
    buf304 = buf282; del buf282  # reuse
    buf305 = buf303; del buf303  # reuse
    cpp_fused_native_batch_norm_backward_59(c_void_p(buf305.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(mm_25.data_ptr()), c_void_p(unsqueeze_153.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()))
    del mm_25
    del primals_103
    del squeeze_88
    del unsqueeze_153
    buf306 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (512, 392), (1, 512), 0), view_152, out=buf306)
    del view_152
    buf307 = reinterpret_tensor(buf292, (392, 256), (256, 1), 0); del buf292  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (392, 512), (512, 1), 0), permute_319, out=buf307)
    del permute_319
    buf308 = empty((256, ), device='cpu', dtype=torch.float32)
    buf309 = empty((256, ), device='cpu', dtype=torch.float32)
    buf310 = reinterpret_tensor(buf291, (392, 256), (256, 1), 0); del buf291  # reuse
    buf312 = buf310; del buf310  # reuse
    buf311 = buf309; del buf309  # reuse
    cpp_fused_native_batch_norm_backward_60(c_void_p(buf312.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(mm_24.data_ptr()), c_void_p(unsqueeze_157.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf308.data_ptr()))
    del mm_24
    del primals_100
    del squeeze_85
    del unsqueeze_157
    buf313 = empty((256, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (256, 392), (1, 256), 0), view_148, out=buf313)
    del view_148
    buf314 = buf304; del buf304  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (392, 256), (256, 1), 0), permute_323, out=buf314)
    del permute_323
    buf315 = empty((512, ), device='cpu', dtype=torch.float32)
    buf316 = empty((512, ), device='cpu', dtype=torch.float32)
    buf317 = empty((512, ), device='cpu', dtype=torch.float32)
    buf318 = buf314; del buf314  # reuse
    cpp_fused_native_batch_norm_backward_61(c_void_p(buf318.data_ptr()), c_void_p(view_147.data_ptr()), c_void_p(mm_23.data_ptr()), c_void_p(unsqueeze_161.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    del mm_23
    del primals_97
    del squeeze_82
    del unsqueeze_161
    del view_147
    buf319 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (512, 392), (1, 512), 0), view_144, out=buf319)
    del view_144
    buf320 = buf312; del buf312  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (392, 512), (512, 1), 0), permute_327, out=buf320)
    del permute_327
    buf321 = buf249; del buf249  # reuse
    buf322 = empty((256, ), device='cpu', dtype=torch.float32)
    buf323 = empty((256, ), device='cpu', dtype=torch.float32)
    buf324 = empty((256, ), device='cpu', dtype=torch.float32)
    buf325 = empty((392, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_batch_norm_backward_62(c_void_p(buf321.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(mm_22.data_ptr()), c_void_p(unsqueeze_165.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del buf272
    del buf284
    del buf307
    del mm_22
    del primals_94
    del squeeze_79
    del unsqueeze_165
    buf326 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (256, 392), (1, 256), 0), view_140, out=buf326)
    del view_140
    buf327 = buf320; del buf320  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (392, 256), (256, 1), 0), permute_331, out=buf327)
    del permute_331
    buf328 = reinterpret_tensor(buf325, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf325  # reuse
    cpp_fused_clone_63(c_void_p(view_139.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    del view_139
    buf329 = reinterpret_tensor(buf327, (64, 49, 32), (1568, 32, 1), 0); del buf327  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_334, reinterpret_tensor(buf328, (64, 49, 32), (1568, 32, 1), 0), out=buf329)
    del permute_334
    buf330 = reinterpret_tensor(buf299, (64, 49, 49), (2401, 49, 1), 0); del buf299  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf328, (64, 49, 32), (1568, 32, 1), 0), permute_335, out=buf330)
    del permute_335
    buf331 = buf294; del buf294  # reuse
    buf332 = buf295; del buf295  # reuse
    buf333 = empty((8, 49), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_new_zeros_sum_64(c_void_p(buf330.data_ptr()), c_void_p(alias_22.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    aten.index_put_(buf333, [None, primals_214], reinterpret_tensor(buf332, (8, 49, 49), (2401, 49, 1), 0), True)
    del buf332
    del primals_214
    buf336 = reinterpret_tensor(buf330, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf330  # reuse
    cpp_fused__softmax_backward_data_mul_65(c_void_p(buf336.data_ptr()), c_void_p(alias_22.data_ptr()), c_void_p(buf331.data_ptr()))
    del alias_22
    buf337 = reinterpret_tensor(buf301, (64, 16, 49), (784, 49, 1), 0); del buf301  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_336, reinterpret_tensor(buf336, (64, 49, 49), (2401, 49, 1), 0), out=buf337)
    del permute_336
    buf338 = reinterpret_tensor(buf300, (64, 49, 16), (784, 16, 1), 0); del buf300  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (64, 49, 49), (2401, 49, 1), 0), permute_337, out=buf338)
    del permute_337
    buf339 = buf316; del buf316  # reuse
    buf340 = empty((512, ), device='cpu', dtype=torch.float32)
    buf341 = buf318; del buf318  # reuse
    buf342 = buf340; del buf340  # reuse
    cpp_fused_native_batch_norm_backward_66(c_void_p(buf342.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(mm_21.data_ptr()), c_void_p(unsqueeze_169.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()))
    del mm_21
    del primals_91
    del squeeze_76
    del unsqueeze_169
    buf343 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (512, 392), (1, 512), 0), view_128, out=buf343)
    del view_128
    buf344 = reinterpret_tensor(buf329, (392, 256), (256, 1), 0); del buf329  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (392, 512), (512, 1), 0), permute_343, out=buf344)
    del permute_343
    buf345 = buf323; del buf323  # reuse
    buf346 = empty((256, ), device='cpu', dtype=torch.float32)
    buf347 = empty((256, ), device='cpu', dtype=torch.float32)
    buf348 = reinterpret_tensor(buf328, (392, 256), (256, 1), 0); del buf328  # reuse
    cpp_fused_native_batch_norm_backward_67(c_void_p(buf321.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(mm_20.data_ptr()), c_void_p(unsqueeze_173.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    del mm_20
    del primals_88
    del squeeze_73
    del unsqueeze_173
    buf349 = empty((256, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf348, (256, 392), (1, 256), 0), view_124, out=buf349)
    del view_124
    buf350 = buf341; del buf341  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf348, (392, 256), (256, 1), 0), permute_347, out=buf350)
    del permute_347
    buf351 = empty((512, ), device='cpu', dtype=torch.float32)
    buf352 = empty((512, ), device='cpu', dtype=torch.float32)
    buf353 = empty((512, ), device='cpu', dtype=torch.float32)
    buf354 = buf350; del buf350  # reuse
    cpp_fused_native_batch_norm_backward_68(c_void_p(buf354.data_ptr()), c_void_p(view_123.data_ptr()), c_void_p(mm_19.data_ptr()), c_void_p(unsqueeze_177.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    del buf352
    del mm_19
    del primals_85
    del squeeze_70
    del unsqueeze_177
    del view_123
    buf355 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf354, (512, 392), (1, 512), 0), view_120, out=buf355)
    del view_120
    buf356 = buf348; del buf348  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf354, (392, 512), (512, 1), 0), permute_351, out=buf356)
    del permute_351
    buf357 = buf346; del buf346  # reuse
    buf358 = empty((256, ), device='cpu', dtype=torch.float32)
    buf359 = reinterpret_tensor(buf321, (392, 256), (256, 1), 0); del buf321  # reuse
    buf360 = buf358; del buf358  # reuse
    cpp_fused_native_batch_norm_backward_69(c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(mm_18.data_ptr()), c_void_p(unsqueeze_181.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf357.data_ptr()))
    del buf344
    del mm_18
    del primals_82
    del squeeze_67
    del unsqueeze_181
    buf361 = empty((256, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (256, 392), (1, 256), 0), view_116, out=buf361)
    del view_116
    buf362 = buf354; del buf354  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (392, 256), (256, 1), 0), permute_355, out=buf362)
    del permute_355
    buf363 = empty((8, 8, 49, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_70(c_void_p(view_115.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    del view_115
    buf364 = empty((64, 196, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_358, reinterpret_tensor(buf363, (64, 49, 64), (3136, 64, 1), 0), out=buf364)
    del permute_358
    buf365 = empty((64, 49, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf363, (64, 49, 64), (3136, 64, 1), 0), permute_359, out=buf365)
    del permute_359
    buf366 = buf331; del buf331  # reuse
    buf367 = empty((1, 8, 49, 196), device='cpu', dtype=torch.float32)
    buf368 = empty((8, 196), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_new_zeros_sum_71(c_void_p(buf365.data_ptr()), c_void_p(alias_23.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    aten.index_put_(buf368, [None, primals_213], reinterpret_tensor(buf367, (8, 49, 196), (9604, 196, 1), 0), True)
    del buf367
    del primals_213
    buf371 = reinterpret_tensor(buf365, (8, 8, 49, 196), (76832, 9604, 196, 1), 0); del buf365  # reuse
    cpp_fused__softmax_backward_data_mul_72(c_void_p(buf371.data_ptr()), c_void_p(alias_23.data_ptr()), c_void_p(buf366.data_ptr()))
    del alias_23
    del buf366
    buf372 = reinterpret_tensor(buf363, (64, 16, 196), (3136, 196, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_360, reinterpret_tensor(buf371, (64, 49, 196), (9604, 196, 1), 0), out=buf372)
    del permute_360
    buf373 = buf338; del buf338  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf371, (64, 49, 196), (9604, 196, 1), 0), permute_361, out=buf373)
    del buf371
    del permute_361
    buf374 = empty((128, ), device='cpu', dtype=torch.float32)
    buf375 = empty((128, ), device='cpu', dtype=torch.float32)
    buf376 = empty((128, ), device='cpu', dtype=torch.float32)
    buf377 = reinterpret_tensor(buf337, (392, 128), (128, 1), 0); del buf337  # reuse
    cpp_fused_native_batch_norm_backward_73(c_void_p(buf373.data_ptr()), c_void_p(mm_17.data_ptr()), c_void_p(unsqueeze_185.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del mm_17
    del primals_79
    del squeeze_64
    del unsqueeze_185
    buf378 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (128, 392), (1, 128), 0), view_104, out=buf378)
    del view_104
    buf379 = reinterpret_tensor(buf373, (392, 128), (128, 1), 0); del buf373  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (392, 128), (128, 1), 0), permute_365, out=buf379)
    del buf377
    del permute_365
    buf380 = empty((640, ), device='cpu', dtype=torch.float32)
    buf381 = empty((640, ), device='cpu', dtype=torch.float32)
    buf382 = empty((640, ), device='cpu', dtype=torch.float32)
    buf383 = empty((1568, 640), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_74(c_void_p(buf372.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(mm_16.data_ptr()), c_void_p(unsqueeze_189.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    del buf364
    del buf381
    del mm_16
    del primals_76
    del squeeze_61
    del unsqueeze_189
    buf384 = empty((640, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf383, (640, 1568), (1, 640), 0), view_97, out=buf384)
    del view_97
    buf385 = reinterpret_tensor(buf372, (1568, 128), (128, 1), 0); del buf372  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf383, (1568, 640), (640, 1), 0), permute_371, out=buf385)
    del buf383
    del permute_371
    buf386 = buf375; del buf375  # reuse
    buf387 = empty((128, ), device='cpu', dtype=torch.float32)
    buf388 = empty((128, ), device='cpu', dtype=torch.float32)
    buf389 = reinterpret_tensor(buf362, (1568, 128), (128, 1), 0); del buf362  # reuse
    cpp_fused_native_batch_norm_backward_75(c_void_p(buf379.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(mm_15.data_ptr()), c_void_p(unsqueeze_193.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del mm_15
    del primals_73
    del squeeze_58
    del unsqueeze_193
    buf390 = buf194; del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (128, 1568), (1, 128), 0), view_93, out=buf390)
    del view_93
    buf391 = reinterpret_tensor(buf179, (1568, 256), (256, 1), 0); del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (1568, 128), (128, 1), 0), permute_375, out=buf391)
    del permute_375
    buf392 = empty((256, ), device='cpu', dtype=torch.float32)
    buf393 = empty((256, ), device='cpu', dtype=torch.float32)
    buf394 = empty((256, ), device='cpu', dtype=torch.float32)
    buf395 = buf391; del buf391  # reuse
    cpp_fused_native_batch_norm_backward_76(c_void_p(buf395.data_ptr()), c_void_p(view_92.data_ptr()), c_void_p(mm_14.data_ptr()), c_void_p(unsqueeze_197.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    del mm_14
    del primals_70
    del squeeze_55
    del unsqueeze_197
    del view_92
    buf396 = reinterpret_tensor(buf192, (256, 128), (128, 1), 0); del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (256, 1568), (1, 256), 0), view_89, out=buf396)
    del view_89
    buf397 = buf389; del buf389  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (1568, 256), (256, 1), 0), permute_379, out=buf397)
    del permute_379
    buf398 = buf387; del buf387  # reuse
    buf399 = empty((128, ), device='cpu', dtype=torch.float32)
    buf400 = empty((1568, 128), device='cpu', dtype=torch.float32)
    buf401 = buf399; del buf399  # reuse
    cpp_fused_native_batch_norm_backward_77(c_void_p(buf401.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(mm_13.data_ptr()), c_void_p(unsqueeze_201.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf400.data_ptr()))
    del mm_13
    del primals_67
    del squeeze_52
    del unsqueeze_201
    buf402 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf400, (128, 1568), (1, 128), 0), view_85, out=buf402)
    del view_85
    buf403 = empty((1568, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf400, (1568, 128), (128, 1), 0), permute_383, out=buf403)
    del permute_383
    buf404 = reinterpret_tensor(buf400, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf400  # reuse
    cpp_fused_clone_78(c_void_p(view_84.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del view_84
    buf405 = reinterpret_tensor(buf403, (32, 196, 32), (6272, 32, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_386, reinterpret_tensor(buf404, (32, 196, 32), (6272, 32, 1), 0), out=buf405)
    del permute_386
    buf406 = empty((32, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf404, (32, 196, 32), (6272, 32, 1), 0), permute_387, out=buf406)
    del permute_387
    buf407 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf408 = reinterpret_tensor(buf336, (1, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf336  # reuse
    buf409 = empty((4, 196), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_79(c_void_p(buf406.data_ptr()), c_void_p(alias_24.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    aten.index_put_(buf409, [None, primals_212], reinterpret_tensor(buf408, (4, 196, 196), (38416, 196, 1), 0), True)
    del primals_212
    buf412 = reinterpret_tensor(buf406, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf406  # reuse
    cpp_fused__softmax_backward_data_mul_80(c_void_p(buf412.data_ptr()), c_void_p(alias_24.data_ptr()), c_void_p(buf407.data_ptr()))
    del alias_24
    buf413 = reinterpret_tensor(buf359, (32, 16, 196), (3136, 196, 1), 0); del buf359  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_388, reinterpret_tensor(buf412, (32, 196, 196), (38416, 196, 1), 0), out=buf413)
    del permute_388
    buf414 = reinterpret_tensor(buf356, (32, 196, 16), (3136, 16, 1), 0); del buf356  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf412, (32, 196, 196), (38416, 196, 1), 0), permute_389, out=buf414)
    del permute_389
    buf415 = buf393; del buf393  # reuse
    buf416 = empty((256, ), device='cpu', dtype=torch.float32)
    buf417 = buf395; del buf395  # reuse
    buf418 = buf416; del buf416  # reuse
    cpp_fused_native_batch_norm_backward_81(c_void_p(buf418.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(mm_12.data_ptr()), c_void_p(unsqueeze_205.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()))
    del mm_12
    del primals_64
    del squeeze_49
    del unsqueeze_205
    buf419 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (256, 1568), (1, 256), 0), view_73, out=buf419)
    del view_73
    buf420 = reinterpret_tensor(buf405, (1568, 128), (128, 1), 0); del buf405  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (1568, 256), (256, 1), 0), permute_395, out=buf420)
    del permute_395
    buf421 = empty((128, ), device='cpu', dtype=torch.float32)
    buf422 = empty((128, ), device='cpu', dtype=torch.float32)
    buf423 = reinterpret_tensor(buf404, (1568, 128), (128, 1), 0); del buf404  # reuse
    buf425 = buf423; del buf423  # reuse
    buf424 = buf422; del buf422  # reuse
    cpp_fused_native_batch_norm_backward_82(c_void_p(buf425.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(mm_11.data_ptr()), c_void_p(unsqueeze_209.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf421.data_ptr()))
    del mm_11
    del primals_61
    del squeeze_46
    del unsqueeze_209
    buf426 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf425, (128, 1568), (1, 128), 0), view_69, out=buf426)
    del view_69
    buf427 = buf417; del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf425, (1568, 128), (128, 1), 0), permute_399, out=buf427)
    del permute_399
    buf428 = empty((256, ), device='cpu', dtype=torch.float32)
    buf429 = empty((256, ), device='cpu', dtype=torch.float32)
    buf430 = empty((256, ), device='cpu', dtype=torch.float32)
    buf431 = buf427; del buf427  # reuse
    cpp_fused_native_batch_norm_backward_83(c_void_p(buf431.data_ptr()), c_void_p(view_68.data_ptr()), c_void_p(mm_10.data_ptr()), c_void_p(unsqueeze_213.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()))
    del mm_10
    del primals_58
    del squeeze_43
    del unsqueeze_213
    del view_68
    buf432 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (256, 1568), (1, 256), 0), view_65, out=buf432)
    del view_65
    buf433 = buf425; del buf425  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (1568, 256), (256, 1), 0), permute_403, out=buf433)
    del permute_403
    buf434 = reinterpret_tensor(buf385, (8, 196, 128), (25088, 128, 1), 0); del buf385  # reuse
    buf435 = empty((128, ), device='cpu', dtype=torch.float32)
    buf436 = empty((128, ), device='cpu', dtype=torch.float32)
    buf437 = empty((128, ), device='cpu', dtype=torch.float32)
    buf438 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_batch_norm_backward_84(c_void_p(buf434.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(mm_9.data_ptr()), c_void_p(unsqueeze_217.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    del buf379
    del mm_9
    del primals_55
    del squeeze_40
    del unsqueeze_217
    buf439 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf438, (128, 1568), (1, 128), 0), view_61, out=buf439)
    del view_61
    buf440 = buf433; del buf433  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf438, (1568, 128), (128, 1), 0), permute_407, out=buf440)
    del permute_407
    buf441 = reinterpret_tensor(buf438, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf438  # reuse
    cpp_fused_clone_85(c_void_p(view_60.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    del view_60
    buf442 = reinterpret_tensor(buf440, (32, 196, 32), (6272, 32, 1), 0); del buf440  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_410, reinterpret_tensor(buf441, (32, 196, 32), (6272, 32, 1), 0), out=buf442)
    del permute_410
    buf443 = reinterpret_tensor(buf412, (32, 196, 196), (38416, 196, 1), 0); del buf412  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf441, (32, 196, 32), (6272, 32, 1), 0), permute_411, out=buf443)
    del permute_411
    buf444 = buf407; del buf407  # reuse
    buf445 = buf408; del buf408  # reuse
    buf446 = empty((4, 196), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_86(c_void_p(buf443.data_ptr()), c_void_p(alias_25.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()))
    aten.index_put_(buf446, [None, primals_211], reinterpret_tensor(buf445, (4, 196, 196), (38416, 196, 1), 0), True)
    del primals_211
    buf449 = reinterpret_tensor(buf443, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf443  # reuse
    cpp_fused__softmax_backward_data_mul_87(c_void_p(buf449.data_ptr()), c_void_p(alias_25.data_ptr()), c_void_p(buf444.data_ptr()))
    del alias_25
    buf450 = reinterpret_tensor(buf414, (32, 16, 196), (3136, 196, 1), 0); del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_412, reinterpret_tensor(buf449, (32, 196, 196), (38416, 196, 1), 0), out=buf450)
    del permute_412
    buf451 = reinterpret_tensor(buf413, (32, 196, 16), (3136, 16, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf449, (32, 196, 196), (38416, 196, 1), 0), permute_413, out=buf451)
    del permute_413
    buf452 = buf429; del buf429  # reuse
    buf453 = empty((256, ), device='cpu', dtype=torch.float32)
    buf454 = buf431; del buf431  # reuse
    buf455 = buf453; del buf453  # reuse
    cpp_fused_native_batch_norm_backward_88(c_void_p(buf455.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(mm_8.data_ptr()), c_void_p(unsqueeze_221.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()))
    del mm_8
    del primals_52
    del squeeze_37
    del unsqueeze_221
    buf456 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (256, 1568), (1, 256), 0), view_49, out=buf456)
    del view_49
    buf457 = reinterpret_tensor(buf442, (1568, 128), (128, 1), 0); del buf442  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (1568, 256), (256, 1), 0), permute_419, out=buf457)
    del permute_419
    buf458 = buf436; del buf436  # reuse
    buf459 = empty((128, ), device='cpu', dtype=torch.float32)
    buf460 = empty((128, ), device='cpu', dtype=torch.float32)
    buf461 = reinterpret_tensor(buf441, (1568, 128), (128, 1), 0); del buf441  # reuse
    cpp_fused_native_batch_norm_backward_89(c_void_p(buf434.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(mm_7.data_ptr()), c_void_p(unsqueeze_225.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()))
    del mm_7
    del primals_49
    del squeeze_34
    del unsqueeze_225
    buf462 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (128, 1568), (1, 128), 0), view_45, out=buf462)
    del view_45
    buf463 = buf454; del buf454  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (1568, 128), (128, 1), 0), permute_423, out=buf463)
    del permute_423
    buf464 = empty((256, ), device='cpu', dtype=torch.float32)
    buf465 = empty((256, ), device='cpu', dtype=torch.float32)
    buf466 = empty((256, ), device='cpu', dtype=torch.float32)
    buf467 = buf463; del buf463  # reuse
    cpp_fused_native_batch_norm_backward_90(c_void_p(buf467.data_ptr()), c_void_p(view_44.data_ptr()), c_void_p(mm_6.data_ptr()), c_void_p(unsqueeze_229.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    del mm_6
    del primals_46
    del squeeze_31
    del unsqueeze_229
    del view_44
    buf468 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (256, 1568), (1, 256), 0), view_41, out=buf468)
    del view_41
    buf469 = buf461; del buf461  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (1568, 256), (256, 1), 0), permute_427, out=buf469)
    del permute_427
    buf470 = buf459; del buf459  # reuse
    buf471 = empty((128, ), device='cpu', dtype=torch.float32)
    buf472 = buf420; del buf420  # reuse
    buf473 = buf471; del buf471  # reuse
    cpp_fused_native_batch_norm_backward_91(c_void_p(buf473.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(unsqueeze_233.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf472.data_ptr()))
    del mm_5
    del primals_43
    del squeeze_28
    del unsqueeze_233
    buf474 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf472, (128, 1568), (1, 128), 0), view_37, out=buf474)
    del view_37
    buf475 = buf397; del buf397  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf472, (1568, 128), (128, 1), 0), permute_431, out=buf475)
    del permute_431
    buf476 = reinterpret_tensor(buf472, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf472  # reuse
    cpp_fused_clone_92(c_void_p(view_36.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()))
    del view_36
    buf477 = reinterpret_tensor(buf475, (32, 196, 32), (6272, 32, 1), 0); del buf475  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_434, reinterpret_tensor(buf476, (32, 196, 32), (6272, 32, 1), 0), out=buf477)
    del permute_434
    buf478 = reinterpret_tensor(buf449, (32, 196, 196), (38416, 196, 1), 0); del buf449  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf476, (32, 196, 32), (6272, 32, 1), 0), permute_435, out=buf478)
    del permute_435
    buf479 = buf444; del buf444  # reuse
    buf480 = buf445; del buf445  # reuse
    buf481 = empty((4, 196), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_index_put_new_zeros_sum_93(c_void_p(buf478.data_ptr()), c_void_p(alias_26.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()))
    aten.index_put_(buf481, [None, primals_210], reinterpret_tensor(buf480, (4, 196, 196), (38416, 196, 1), 0), True)
    del primals_210
    buf484 = reinterpret_tensor(buf478, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf478  # reuse
    cpp_fused__softmax_backward_data_mul_94(c_void_p(buf484.data_ptr()), c_void_p(alias_26.data_ptr()), c_void_p(buf479.data_ptr()))
    del alias_26
    buf485 = reinterpret_tensor(buf451, (32, 16, 196), (3136, 196, 1), 0); del buf451  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_436, reinterpret_tensor(buf484, (32, 196, 196), (38416, 196, 1), 0), out=buf485)
    del permute_436
    buf486 = reinterpret_tensor(buf450, (32, 196, 16), (3136, 16, 1), 0); del buf450  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf484, (32, 196, 196), (38416, 196, 1), 0), permute_437, out=buf486)
    del permute_437
    buf487 = buf465; del buf465  # reuse
    buf488 = empty((256, ), device='cpu', dtype=torch.float32)
    buf489 = buf467; del buf467  # reuse
    buf490 = buf488; del buf488  # reuse
    cpp_fused_native_batch_norm_backward_95(c_void_p(buf490.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(mm_4.data_ptr()), c_void_p(unsqueeze_237.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf489.data_ptr()))
    del mm_4
    del primals_40
    del squeeze_25
    del unsqueeze_237
    buf491 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf489, (256, 1568), (1, 256), 0), view_25, out=buf491)
    del view_25
    buf492 = reinterpret_tensor(buf477, (1568, 128), (128, 1), 0); del buf477  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf489, (1568, 256), (256, 1), 0), permute_443, out=buf492)
    del permute_443
    buf493 = empty((128, ), device='cpu', dtype=torch.float32)
    buf494 = empty((128, ), device='cpu', dtype=torch.float32)
    buf495 = reinterpret_tensor(buf476, (1568, 128), (128, 1), 0); del buf476  # reuse
    buf497 = buf495; del buf495  # reuse
    buf496 = buf494; del buf494  # reuse
    cpp_fused_native_batch_norm_backward_96(c_void_p(buf497.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(unsqueeze_241.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf493.data_ptr()))
    del mm_3
    del primals_37
    del squeeze_22
    del unsqueeze_241
    buf498 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf497, (128, 1568), (1, 128), 0), view_21, out=buf498)
    del view_21
    buf499 = buf489; del buf489  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf497, (1568, 128), (128, 1), 0), permute_447, out=buf499)
    del permute_447
    buf500 = empty((256, ), device='cpu', dtype=torch.float32)
    buf501 = empty((256, ), device='cpu', dtype=torch.float32)
    buf502 = empty((256, ), device='cpu', dtype=torch.float32)
    buf503 = buf499; del buf499  # reuse
    cpp_fused_native_batch_norm_backward_97(c_void_p(buf503.data_ptr()), c_void_p(view_20.data_ptr()), c_void_p(mm_2.data_ptr()), c_void_p(unsqueeze_245.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()))
    del mm_2
    del primals_34
    del squeeze_19
    del unsqueeze_245
    del view_20
    buf504 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (256, 1568), (1, 256), 0), view_17, out=buf504)
    del view_17
    buf505 = buf497; del buf497  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (1568, 256), (256, 1), 0), permute_451, out=buf505)
    del permute_451
    buf506 = buf434; del buf434  # reuse
    buf507 = empty((128, ), device='cpu', dtype=torch.float32)
    buf508 = empty((128, ), device='cpu', dtype=torch.float32)
    buf509 = empty((128, ), device='cpu', dtype=torch.float32)
    buf510 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_batch_norm_backward_98(c_void_p(buf506.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(mm_1.data_ptr()), c_void_p(unsqueeze_249.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()))
    del buf457
    del buf469
    del buf492
    del mm_1
    del primals_31
    del squeeze_16
    del unsqueeze_249
    buf511 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf510, (128, 1568), (1, 128), 0), view_13, out=buf511)
    del view_13
    buf512 = buf505; del buf505  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf510, (1568, 128), (128, 1), 0), permute_455, out=buf512)
    del permute_455
    buf513 = reinterpret_tensor(buf510, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf510  # reuse
    cpp_fused_clone_99(c_void_p(view_12.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()))
    del view_12
    buf514 = reinterpret_tensor(buf512, (32, 196, 32), (6272, 32, 1), 0); del buf512  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_458, reinterpret_tensor(buf513, (32, 196, 32), (6272, 32, 1), 0), out=buf514)
    del permute_458
    buf515 = reinterpret_tensor(buf484, (32, 196, 196), (38416, 196, 1), 0); del buf484  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf513, (32, 196, 32), (6272, 32, 1), 0), permute_459, out=buf515)
    del buf513
    del permute_459
    buf516 = buf479; del buf479  # reuse
    buf517 = buf480; del buf480  # reuse
    buf518 = empty((4, 196), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_new_zeros_sum_100(c_void_p(buf515.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    aten.index_put_(buf518, [None, primals_209], reinterpret_tensor(buf517, (4, 196, 196), (38416, 196, 1), 0), True)
    del buf517
    del primals_209
    buf521 = reinterpret_tensor(buf515, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf515  # reuse
    cpp_fused__softmax_backward_data_mul_101(c_void_p(buf521.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(buf516.data_ptr()))
    del alias_27
    del buf516
    buf522 = reinterpret_tensor(buf486, (32, 16, 196), (3136, 196, 1), 0); del buf486  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_460, reinterpret_tensor(buf521, (32, 196, 196), (38416, 196, 1), 0), out=buf522)
    del permute_460
    buf523 = reinterpret_tensor(buf485, (32, 196, 16), (3136, 16, 1), 0); del buf485  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf521, (32, 196, 196), (38416, 196, 1), 0), permute_461, out=buf523)
    del buf521
    del permute_461
    buf524 = buf501; del buf501  # reuse
    buf525 = empty((256, ), device='cpu', dtype=torch.float32)
    buf526 = buf503; del buf503  # reuse
    buf527 = buf525; del buf525  # reuse
    cpp_fused_native_batch_norm_backward_102(c_void_p(buf527.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(mm.data_ptr()), c_void_p(unsqueeze_253.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf526.data_ptr()))
    del buf522
    del buf523
    del mm
    del primals_28
    del squeeze_13
    del unsqueeze_253
    buf528 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf526, (256, 1568), (1, 256), 0), view_1, out=buf528)
    del view_1
    buf529 = reinterpret_tensor(buf514, (1568, 128), (128, 1), 0); del buf514  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf526, (1568, 256), (256, 1), 0), permute_467, out=buf529)
    del buf526
    del permute_467
    buf530 = buf508; del buf508  # reuse
    buf531 = empty((128, ), device='cpu', dtype=torch.float32)
    buf532 = empty((128, ), device='cpu', dtype=torch.float32)
    buf533 = reinterpret_tensor(buf506, (8, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf506  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_103(c_void_p(buf533.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_259.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    del buf529
    del buf531
    del convolution_3
    del primals_25
    del squeeze_10
    del unsqueeze_259
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf534 = aten.convolution_backward(buf533, div_2, primals_24, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf533
    del div_2
    del primals_24
    buf535 = buf534[0]
    buf536 = buf534[1]
    del buf534
    buf537 = empty((64, ), device='cpu', dtype=torch.float32)
    buf538 = empty((64, ), device='cpu', dtype=torch.float32)
    buf539 = empty((64, ), device='cpu', dtype=torch.float32)
    buf540 = buf535; del buf535  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_104(c_void_p(buf540.data_ptr()), c_void_p(add_16.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_271.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()))
    del add_16
    del buf538
    del convolution_2
    del primals_22
    del squeeze_7
    del unsqueeze_271
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf541 = aten.convolution_backward(buf540, div_1, primals_21, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf540
    del div_1
    del primals_21
    buf542 = buf541[0]
    buf543 = buf541[1]
    del buf541
    buf544 = empty((32, ), device='cpu', dtype=torch.float32)
    buf545 = empty((32, ), device='cpu', dtype=torch.float32)
    buf546 = empty((32, ), device='cpu', dtype=torch.float32)
    buf547 = buf542; del buf542  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_105(c_void_p(buf547.data_ptr()), c_void_p(add_10.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_283.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()))
    del add_10
    del buf545
    del convolution_1
    del primals_19
    del squeeze_4
    del unsqueeze_283
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf548 = aten.convolution_backward(buf547, div, primals_18, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf547
    del div
    del primals_18
    buf549 = buf548[0]
    buf550 = buf548[1]
    del buf548
    buf551 = empty((16, ), device='cpu', dtype=torch.float32)
    buf552 = empty((16, ), device='cpu', dtype=torch.float32)
    buf553 = empty((16, ), device='cpu', dtype=torch.float32)
    buf554 = buf549; del buf549  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_106(c_void_p(buf554.data_ptr()), c_void_p(add_4.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_295.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()))
    del add_4
    del buf552
    del convolution
    del primals_16
    del squeeze_1
    del unsqueeze_295
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf555 = aten.convolution_backward(buf554, primals_415, primals_15, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf554
    del primals_15
    del primals_415
    buf556 = buf555[1]
    return (buf518, buf481, buf446, buf409, buf368, buf333, buf296, buf261, buf224, buf183, buf147, buf111, buf75, buf39, buf556, buf553, buf551, buf550, buf546, buf544, buf543, buf539, buf537, buf536, buf532, buf530, reinterpret_tensor(buf528, (256, 128), (128, 1), 0), buf527, buf524, reinterpret_tensor(buf511, (128, 128), (128, 1), 0), buf509, buf507, reinterpret_tensor(buf504, (256, 128), (128, 1), 0), buf502, buf500, reinterpret_tensor(buf498, (128, 256), (256, 1), 0), buf496, buf493, reinterpret_tensor(buf491, (256, 128), (128, 1), 0), buf490, buf487, reinterpret_tensor(buf474, (128, 128), (128, 1), 0), buf473, buf470, reinterpret_tensor(buf468, (256, 128), (128, 1), 0), buf466, buf464, reinterpret_tensor(buf462, (128, 256), (256, 1), 0), buf460, buf458, reinterpret_tensor(buf456, (256, 128), (128, 1), 0), buf455, buf452, reinterpret_tensor(buf439, (128, 128), (128, 1), 0), buf437, buf435, reinterpret_tensor(buf432, (256, 128), (128, 1), 0), buf430, buf428, reinterpret_tensor(buf426, (128, 256), (256, 1), 0), buf424, buf421, reinterpret_tensor(buf419, (256, 128), (128, 1), 0), buf418, buf415, reinterpret_tensor(buf402, (128, 128), (128, 1), 0), buf401, buf398, reinterpret_tensor(buf396, (256, 128), (128, 1), 0), buf394, buf392, reinterpret_tensor(buf390, (128, 256), (256, 1), 0), buf388, buf386, reinterpret_tensor(buf384, (640, 128), (128, 1), 0), buf382, buf380, reinterpret_tensor(buf378, (128, 128), (128, 1), 0), buf376, buf374, reinterpret_tensor(buf361, (256, 512), (512, 1), 0), buf360, buf357, reinterpret_tensor(buf355, (512, 256), (256, 1), 0), buf353, buf351, reinterpret_tensor(buf349, (256, 512), (512, 1), 0), buf347, buf345, reinterpret_tensor(buf343, (512, 256), (256, 1), 0), buf342, buf339, reinterpret_tensor(buf326, (256, 256), (256, 1), 0), buf324, buf322, reinterpret_tensor(buf319, (512, 256), (256, 1), 0), buf317, buf315, reinterpret_tensor(buf313, (256, 512), (512, 1), 0), buf311, buf308, reinterpret_tensor(buf306, (512, 256), (256, 1), 0), buf305, buf302, reinterpret_tensor(buf289, (256, 256), (256, 1), 0), buf288, buf285, reinterpret_tensor(buf283, (512, 256), (256, 1), 0), buf281, buf279, reinterpret_tensor(buf277, (256, 512), (512, 1), 0), buf275, buf273, reinterpret_tensor(buf271, (512, 256), (256, 1), 0), buf270, buf267, reinterpret_tensor(buf254, (256, 256), (256, 1), 0), buf252, buf250, reinterpret_tensor(buf247, (512, 256), (256, 1), 0), buf245, buf243, reinterpret_tensor(buf241, (256, 512), (512, 1), 0), buf239, buf236, reinterpret_tensor(buf234, (512, 256), (256, 1), 0), buf233, buf230, reinterpret_tensor(buf217, (256, 256), (256, 1), 0), buf216, buf213, reinterpret_tensor(buf211, (512, 256), (256, 1), 0), buf209, buf207, reinterpret_tensor(buf205, (256, 512), (512, 1), 0), buf203, buf201, reinterpret_tensor(buf199, (1280, 256), (256, 1), 0), buf197, buf195, reinterpret_tensor(buf193, (256, 256), (256, 1), 0), buf191, buf189, reinterpret_tensor(buf176, (384, 1024), (1024, 1), 0), buf174, buf172, reinterpret_tensor(buf170, (768, 384), (384, 1), 0), buf168, buf166, reinterpret_tensor(buf164, (384, 768), (768, 1), 0), buf162, buf160, reinterpret_tensor(buf157, (768, 384), (384, 1), 0), buf156, buf153, reinterpret_tensor(buf140, (384, 384), (384, 1), 0), buf138, buf135, reinterpret_tensor(buf133, (768, 384), (384, 1), 0), buf131, buf129, reinterpret_tensor(buf127, (384, 768), (768, 1), 0), buf126, buf123, reinterpret_tensor(buf121, (768, 384), (384, 1), 0), buf120, buf117, reinterpret_tensor(buf104, (384, 384), (384, 1), 0), buf102, buf100, reinterpret_tensor(buf98, (768, 384), (384, 1), 0), buf96, buf94, reinterpret_tensor(buf92, (384, 768), (768, 1), 0), buf90, buf88, reinterpret_tensor(buf85, (768, 384), (384, 1), 0), buf84, buf81, reinterpret_tensor(buf68, (384, 384), (384, 1), 0), buf66, buf63, reinterpret_tensor(buf61, (768, 384), (384, 1), 0), buf59, buf57, reinterpret_tensor(buf55, (384, 768), (768, 1), 0), buf54, buf51, reinterpret_tensor(buf49, (768, 384), (384, 1), 0), buf48, buf45, reinterpret_tensor(buf32, (384, 384), (384, 1), 0), buf30, buf28, reinterpret_tensor(buf26, (768, 384), (384, 1), 0), buf24, buf22, reinterpret_tensor(buf20, (384, 768), (768, 1), 0), buf18, buf16, buf14, buf12, reinterpret_tensor(buf11, (1000, 384), (384, 1), 0), reinterpret_tensor(buf6, (1000, ), (1, ), 0), buf9, buf7, reinterpret_tensor(buf5, (1000, 384), (384, 1), 0), reinterpret_tensor(buf6, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_15 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_210 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_211 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_212 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_213 = rand_strided((49, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_214 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_215 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_216 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_217 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_218 = rand_strided((16, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_219 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_220 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_221 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_222 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_415 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_4 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_10 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_16 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_12 = rand_strided((8, 196, 128), (25088, 128, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_1 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_2 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_3 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_4 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_36 = rand_strided((8, 196, 128), (25088, 128, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_5 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_6 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_7 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_8 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_60 = rand_strided((8, 196, 128), (25088, 128, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_9 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_10 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_11 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_12 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((8, 196, 128), (25088, 128, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_13 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_14 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_92 = rand_strided((8, 196, 256), (50176, 256, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((1568, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_15 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_97 = rand_strided((1568, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_16 = rand_strided((1568, 640), (640, 1), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((392, 128), (128, 1), device='cpu', dtype=torch.float32)
    mm_17 = rand_strided((392, 128), (128, 1), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((8, 49, 512), (25088, 512, 1), device='cpu', dtype=torch.float32)
    view_116 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_18 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_120 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_19 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_123 = rand_strided((8, 49, 512), (25088, 512, 1), device='cpu', dtype=torch.float32)
    view_124 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_20 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_21 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_139 = rand_strided((8, 49, 256), (12544, 256, 1), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_22 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_144 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_23 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_147 = rand_strided((8, 49, 512), (25088, 512, 1), device='cpu', dtype=torch.float32)
    view_148 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_24 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_25 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_163 = rand_strided((8, 49, 256), (12544, 256, 1), device='cpu', dtype=torch.float32)
    view_164 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_26 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_168 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_27 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_171 = rand_strided((8, 49, 512), (25088, 512, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_28 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_29 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_187 = rand_strided((8, 49, 256), (12544, 256, 1), device='cpu', dtype=torch.float32)
    view_188 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_30 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_192 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_31 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_195 = rand_strided((8, 49, 512), (25088, 512, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_32 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_200 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_33 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_211 = rand_strided((8, 49, 256), (12544, 256, 1), device='cpu', dtype=torch.float32)
    view_212 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_34 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_35 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_219 = rand_strided((8, 49, 512), (25088, 512, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((392, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_36 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_224 = rand_strided((392, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_37 = rand_strided((392, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    view_231 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    mm_38 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((8, 16, 1024), (16384, 1024, 1), device='cpu', dtype=torch.float32)
    view_243 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mm_39 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_247 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_40 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_250 = rand_strided((8, 16, 768), (12288, 768, 1), device='cpu', dtype=torch.float32)
    view_251 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mm_41 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_255 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_42 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_266 = rand_strided((8, 16, 384), (6144, 384, 1), device='cpu', dtype=torch.float32)
    view_267 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_43 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_271 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_44 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_274 = rand_strided((8, 16, 768), (12288, 768, 1), device='cpu', dtype=torch.float32)
    view_275 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mm_45 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_279 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_46 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_290 = rand_strided((8, 16, 384), (6144, 384, 1), device='cpu', dtype=torch.float32)
    view_291 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_47 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_295 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_48 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_298 = rand_strided((8, 16, 768), (12288, 768, 1), device='cpu', dtype=torch.float32)
    view_299 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mm_49 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_303 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_50 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_314 = rand_strided((8, 16, 384), (6144, 384, 1), device='cpu', dtype=torch.float32)
    view_315 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_51 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_319 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_52 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_322 = rand_strided((8, 16, 768), (12288, 768, 1), device='cpu', dtype=torch.float32)
    view_323 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mm_53 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_327 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_54 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_338 = rand_strided((8, 16, 384), (6144, 384, 1), device='cpu', dtype=torch.float32)
    view_339 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_55 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_343 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    mm_56 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_346 = rand_strided((8, 16, 768), (12288, 768, 1), device='cpu', dtype=torch.float32)
    view_347 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mm_57 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 384), (384, 1), device='cpu', dtype=torch.float32)
    clone_81 = rand_strided((8, 384), (384, 1), device='cpu', dtype=torch.float32)
    clone_82 = rand_strided((8, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_121 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_25 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_127 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    unsqueeze_29 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_131 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_33 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_135 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_139 = rand_strided((96, 32, 16), (512, 1, 32), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    permute_140 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_141 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    unsqueeze_37 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_147 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_41 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    unsqueeze_45 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_49 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_162 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_163 = rand_strided((96, 32, 16), (512, 1, 32), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    permute_164 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    unsqueeze_53 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_57 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    unsqueeze_61 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_65 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((96, 32, 16), (512, 1, 32), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    unsqueeze_69 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_73 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    unsqueeze_77 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_203 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_81 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_207 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_210 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_211 = rand_strided((96, 32, 16), (512, 1, 32), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((96, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    unsqueeze_85 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_89 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_223 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    unsqueeze_93 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    unsqueeze_97 = rand_strided((1, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_231 = rand_strided((384, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((128, 49, 16), (784, 1, 49), device='cpu', dtype=torch.float32)
    permute_235 = rand_strided((128, 64, 49), (3136, 1, 64), device='cpu', dtype=torch.float32)
    alias_18 = rand_strided((8, 16, 16, 49), (12544, 1, 784, 16), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((128, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((128, 49, 16), (784, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_101 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_241 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_105 = rand_strided((1, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_247 = rand_strided((1280, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_109 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_251 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    unsqueeze_113 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_117 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_259 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_262 = rand_strided((64, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_263 = rand_strided((64, 32, 49), (1568, 1, 32), device='cpu', dtype=torch.float32)
    alias_19 = rand_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    permute_264 = rand_strided((64, 16, 49), (784, 1, 16), device='cpu', dtype=torch.float32)
    permute_265 = rand_strided((64, 49, 16), (784, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_121 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_271 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_125 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_275 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    unsqueeze_129 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_279 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_133 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_286 = rand_strided((64, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_287 = rand_strided((64, 32, 49), (1568, 1, 32), device='cpu', dtype=torch.float32)
    alias_20 = rand_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((64, 16, 49), (784, 1, 16), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((64, 49, 16), (784, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_137 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_295 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_141 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_299 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    unsqueeze_145 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_303 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_149 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((64, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((64, 32, 49), (1568, 1, 32), device='cpu', dtype=torch.float32)
    alias_21 = rand_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    permute_312 = rand_strided((64, 16, 49), (784, 1, 16), device='cpu', dtype=torch.float32)
    permute_313 = rand_strided((64, 49, 16), (784, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_153 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_319 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_157 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_323 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    unsqueeze_161 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_327 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_165 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_331 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_334 = rand_strided((64, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((64, 32, 49), (1568, 1, 32), device='cpu', dtype=torch.float32)
    alias_22 = rand_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    permute_336 = rand_strided((64, 16, 49), (784, 1, 16), device='cpu', dtype=torch.float32)
    permute_337 = rand_strided((64, 49, 16), (784, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_169 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_343 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_173 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_347 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    unsqueeze_177 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_351 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_181 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((64, 196, 49), (9604, 1, 196), device='cpu', dtype=torch.float32)
    permute_359 = rand_strided((64, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    alias_23 = rand_strided((8, 8, 49, 196), (76832, 1, 1568, 8), device='cpu', dtype=torch.float32)
    permute_360 = rand_strided((64, 16, 49), (784, 1, 16), device='cpu', dtype=torch.float32)
    permute_361 = rand_strided((64, 196, 16), (3136, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_185 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_365 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_189 = rand_strided((1, 640), (640, 1), device='cpu', dtype=torch.float32)
    permute_371 = rand_strided((640, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_193 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_375 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_197 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_379 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_201 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_386 = rand_strided((32, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_387 = rand_strided((32, 32, 196), (6272, 1, 32), device='cpu', dtype=torch.float32)
    alias_24 = rand_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((32, 16, 196), (3136, 1, 16), device='cpu', dtype=torch.float32)
    permute_389 = rand_strided((32, 196, 16), (3136, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_205 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_395 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_209 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_399 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_213 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_403 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_217 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_407 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_410 = rand_strided((32, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_411 = rand_strided((32, 32, 196), (6272, 1, 32), device='cpu', dtype=torch.float32)
    alias_25 = rand_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    permute_412 = rand_strided((32, 16, 196), (3136, 1, 16), device='cpu', dtype=torch.float32)
    permute_413 = rand_strided((32, 196, 16), (3136, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_221 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_419 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_225 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_423 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_229 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_427 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_233 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_431 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((32, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((32, 32, 196), (6272, 1, 32), device='cpu', dtype=torch.float32)
    alias_26 = rand_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    permute_436 = rand_strided((32, 16, 196), (3136, 1, 16), device='cpu', dtype=torch.float32)
    permute_437 = rand_strided((32, 196, 16), (3136, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_237 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_443 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_241 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_447 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    unsqueeze_245 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_451 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_249 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_455 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_458 = rand_strided((32, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_459 = rand_strided((32, 32, 196), (6272, 1, 32), device='cpu', dtype=torch.float32)
    alias_27 = rand_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    permute_460 = rand_strided((32, 16, 196), (3136, 1, 16), device='cpu', dtype=torch.float32)
    permute_461 = rand_strided((32, 196, 16), (3136, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_253 = rand_strided((1, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_467 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    unsqueeze_259 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_271 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_283 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_295 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_79, primals_82, primals_85, primals_88, primals_91, primals_94, primals_97, primals_100, primals_103, primals_106, primals_109, primals_112, primals_115, primals_118, primals_121, primals_124, primals_127, primals_130, primals_133, primals_136, primals_139, primals_142, primals_145, primals_148, primals_151, primals_154, primals_157, primals_160, primals_163, primals_166, primals_169, primals_172, primals_175, primals_178, primals_181, primals_184, primals_187, primals_190, primals_193, primals_196, primals_199, primals_201, primals_205, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_415, convolution, squeeze_1, add_4, div, convolution_1, squeeze_4, add_10, div_1, convolution_2, squeeze_7, add_16, div_2, convolution_3, squeeze_10, view_1, mm, squeeze_13, view_12, view_13, mm_1, squeeze_16, view_17, mm_2, squeeze_19, view_20, view_21, mm_3, squeeze_22, view_25, mm_4, squeeze_25, view_36, view_37, mm_5, squeeze_28, view_41, mm_6, squeeze_31, view_44, view_45, mm_7, squeeze_34, view_49, mm_8, squeeze_37, view_60, view_61, mm_9, squeeze_40, view_65, mm_10, squeeze_43, view_68, view_69, mm_11, squeeze_46, view_73, mm_12, squeeze_49, view_84, view_85, mm_13, squeeze_52, view_89, mm_14, squeeze_55, view_92, view_93, mm_15, squeeze_58, view_97, mm_16, squeeze_61, view_104, mm_17, squeeze_64, view_115, view_116, mm_18, squeeze_67, view_120, mm_19, squeeze_70, view_123, view_124, mm_20, squeeze_73, view_128, mm_21, squeeze_76, view_139, view_140, mm_22, squeeze_79, view_144, mm_23, squeeze_82, view_147, view_148, mm_24, squeeze_85, view_152, mm_25, squeeze_88, view_163, view_164, mm_26, squeeze_91, view_168, mm_27, squeeze_94, view_171, view_172, mm_28, squeeze_97, view_176, mm_29, squeeze_100, view_187, view_188, mm_30, squeeze_103, view_192, mm_31, squeeze_106, view_195, view_196, mm_32, squeeze_109, view_200, mm_33, squeeze_112, view_211, view_212, mm_34, squeeze_115, view_216, mm_35, squeeze_118, view_219, view_220, mm_36, squeeze_121, view_224, mm_37, squeeze_124, view_231, mm_38, squeeze_127, view_242, view_243, mm_39, squeeze_130, view_247, mm_40, squeeze_133, view_250, view_251, mm_41, squeeze_136, view_255, mm_42, squeeze_139, view_266, view_267, mm_43, squeeze_142, view_271, mm_44, squeeze_145, view_274, view_275, mm_45, squeeze_148, view_279, mm_46, squeeze_151, view_290, view_291, mm_47, squeeze_154, view_295, mm_48, squeeze_157, view_298, view_299, mm_49, squeeze_160, view_303, mm_50, squeeze_163, view_314, view_315, mm_51, squeeze_166, view_319, mm_52, squeeze_169, view_322, view_323, mm_53, squeeze_172, view_327, mm_54, squeeze_175, view_338, view_339, mm_55, squeeze_178, view_343, mm_56, squeeze_181, view_346, view_347, mm_57, squeeze_184, mean, clone_81, clone_82, permute_117, permute_121, unsqueeze_25, permute_127, unsqueeze_29, permute_131, unsqueeze_33, permute_135, permute_138, permute_139, alias_14, permute_140, permute_141, unsqueeze_37, permute_147, unsqueeze_41, permute_151, unsqueeze_45, permute_155, unsqueeze_49, permute_159, permute_162, permute_163, alias_15, permute_164, permute_165, unsqueeze_53, permute_171, unsqueeze_57, permute_175, unsqueeze_61, permute_179, unsqueeze_65, permute_183, permute_186, permute_187, alias_16, permute_188, permute_189, unsqueeze_69, permute_195, unsqueeze_73, permute_199, unsqueeze_77, permute_203, unsqueeze_81, permute_207, permute_210, permute_211, alias_17, permute_212, permute_213, unsqueeze_85, permute_219, unsqueeze_89, permute_223, unsqueeze_93, permute_227, unsqueeze_97, permute_231, permute_234, permute_235, alias_18, permute_236, permute_237, unsqueeze_101, permute_241, unsqueeze_105, permute_247, unsqueeze_109, permute_251, unsqueeze_113, permute_255, unsqueeze_117, permute_259, permute_262, permute_263, alias_19, permute_264, permute_265, unsqueeze_121, permute_271, unsqueeze_125, permute_275, unsqueeze_129, permute_279, unsqueeze_133, permute_283, permute_286, permute_287, alias_20, permute_288, permute_289, unsqueeze_137, permute_295, unsqueeze_141, permute_299, unsqueeze_145, permute_303, unsqueeze_149, permute_307, permute_310, permute_311, alias_21, permute_312, permute_313, unsqueeze_153, permute_319, unsqueeze_157, permute_323, unsqueeze_161, permute_327, unsqueeze_165, permute_331, permute_334, permute_335, alias_22, permute_336, permute_337, unsqueeze_169, permute_343, unsqueeze_173, permute_347, unsqueeze_177, permute_351, unsqueeze_181, permute_355, permute_358, permute_359, alias_23, permute_360, permute_361, unsqueeze_185, permute_365, unsqueeze_189, permute_371, unsqueeze_193, permute_375, unsqueeze_197, permute_379, unsqueeze_201, permute_383, permute_386, permute_387, alias_24, permute_388, permute_389, unsqueeze_205, permute_395, unsqueeze_209, permute_399, unsqueeze_213, permute_403, unsqueeze_217, permute_407, permute_410, permute_411, alias_25, permute_412, permute_413, unsqueeze_221, permute_419, unsqueeze_225, permute_423, unsqueeze_229, permute_427, unsqueeze_233, permute_431, permute_434, permute_435, alias_26, permute_436, permute_437, unsqueeze_237, permute_443, unsqueeze_241, permute_447, unsqueeze_245, permute_451, unsqueeze_249, permute_455, permute_458, permute_459, alias_27, permute_460, permute_461, unsqueeze_253, permute_467, unsqueeze_259, unsqueeze_271, unsqueeze_283, unsqueeze_295, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('levit_128', benchmark_compiled_module)
