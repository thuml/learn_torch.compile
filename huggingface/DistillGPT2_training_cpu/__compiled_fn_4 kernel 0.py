
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


cpp_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25681320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(25681320L); x0<static_cast<long>(25681327L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = static_cast<float>(0.0);
                out_ptr0[static_cast<long>(x0)] = tmp0;
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(511L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(1L + x0)];
                    auto tmp1 = static_cast<long>(-100);
                    auto tmp2 = tmp0 != tmp1;
                    auto tmp3 = static_cast<long>(0);
                    auto tmp4 = tmp2 ? tmp0 : tmp3;
                    out_ptr1[static_cast<long>(x0)] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_slice_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const long* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(511L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50257L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(0L)];
                        auto tmp5 = in_ptr3[static_cast<long>(0L)];
                        auto tmp2 = static_cast<int>(-100);
                        auto tmp3 = tmp1 != tmp2;
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp0 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50256L); x1<static_cast<long>(50257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50257L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(0L)];
                        auto tmp5 = in_ptr3[static_cast<long>(0L)];
                        auto tmp2 = static_cast<long>(-100);
                        auto tmp3 = tmp1 != tmp2;
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        auto tmp9 = decltype(tmp0)(tmp0 * tmp8);
                        tmp_acc0 = tmp_acc0 + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (50257L*x0)));
                    auto tmp1 = c10::convert<int>(x0);
                    auto tmp2 = static_cast<int>(511);
                    auto tmp3 = tmp1 < tmp2;
                    auto tmp4 = [&]
                    {
                        auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1 + (50257L*x0)), to_float_mask(tmp3));
                        auto tmp6 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp7 = static_cast<int>(-100);
                        auto tmp8 = tmp6 != tmp7;
                        auto tmp9 = in_ptr2[static_cast<long>(0L)];
                        auto tmp10 = in_ptr3[static_cast<long>(0L)];
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp8 ? tmp11 : tmp12;
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = masked_load(in_ptr5 + static_cast<long>(x1 + (50257L*x0)), to_float_mask(tmp3));
                        auto tmp17 = tmp16.exp();
                        auto tmp18 = out_ptr0[static_cast<long>(x0)];
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp15 - tmp20;
                        return tmp21;
                    }
                    ;
                    auto tmp22 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                    auto tmp23 = static_cast<float>(0.0);
                    auto tmp24 = to_float_mask(tmp3);
                    auto tmp25 = at::vec::Vectorized<float>(tmp23);
                    auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp24);
                    auto tmp27 = tmp0 + tmp26;
                    tmp27.store(out_ptr1 + static_cast<long>(x1 + (50257L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(50256L); x1<static_cast<long>(50257L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x1 + (50257L*x0))];
                    auto tmp1 = c10::convert<long>(x0);
                    auto tmp2 = static_cast<long>(511);
                    auto tmp3 = tmp1 < tmp2;
                    auto tmp4 = [&]
                    {
                        auto tmp5 = in_ptr0[static_cast<long>(x1 + (50257L*x0))];
                        auto tmp6 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp7 = static_cast<long>(-100);
                        auto tmp8 = tmp6 != tmp7;
                        auto tmp9 = in_ptr2[static_cast<long>(0L)];
                        auto tmp10 = in_ptr3[static_cast<long>(0L)];
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp8 ? tmp11 : tmp12;
                        auto tmp14 = decltype(tmp5)(tmp5 * tmp13);
                        auto tmp15 = in_ptr5[static_cast<long>(x1 + (50257L*x0))];
                        auto tmp16 = std::exp(tmp15);
                        auto tmp17 = out_ptr0[static_cast<long>(x0)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = decltype(tmp14)(tmp14 - tmp18);
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                    auto tmp21 = static_cast<float>(0.0);
                    auto tmp22 = tmp3 ? tmp20 : tmp21;
                    auto tmp23 = decltype(tmp0)(tmp0 + tmp22);
                    out_ptr1[static_cast<long>(x1 + (50257L*x0))] = tmp23;
                }
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_native_layer_norm_backward_2 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 - tmp13;
                    auto tmp15 = at::vec::Vectorized<float>(tmp0);
                    auto tmp16 = tmp15 * tmp14;
                    tmp16.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (512L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (512L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (512L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                       float* out_ptr5)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (512L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (512L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (262144L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (262144L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (512L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (32768L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const long* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (768L*x0))];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp9 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                    auto tmp10 = out_ptr2[static_cast<long>(x0)];
                    auto tmp15 = in_ptr5[static_cast<long>(x1 + (768L*x0))];
                    auto tmp20 = in_ptr6[static_cast<long>(x0)];
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                    auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                    auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                    auto tmp16 = c10::convert<float>(tmp15);
                    auto tmp17 = static_cast<float>(1.1111111111111112);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp19 = decltype(tmp14)(tmp14 * tmp18);
                    auto tmp21 = static_cast<long>(-1);
                    auto tmp22 = tmp20 == tmp21;
                    auto tmp23 = static_cast<float>(0.0);
                    auto tmp24 = tmp22 ? tmp23 : tmp19;
                    in_out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp19;
                    out_ptr5[static_cast<long>(x1 + (768L*x0))] = tmp24;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr6 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<bool>(0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 ? tmp2 : tmp0;
                in_out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(38597376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_85, view, view_1, getitem_1, mul, slice_4, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, slice_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, slice_12, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, slice_16, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, slice_20, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, slice_24, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, view_111, sub_20, convert_element_type, permute_33, div_14, permute_35, permute_36, permute_37, permute_38, div_15, permute_39, permute_40, permute_42, permute_43, alias_15, permute_44, permute_45, permute_50, permute_51, div_17, permute_52, permute_53, permute_54, permute_55, div_18, permute_56, permute_57, permute_59, permute_60, alias_17, permute_61, permute_62, permute_67, permute_68, div_20, permute_69, permute_70, permute_71, permute_72, div_21, permute_73, permute_74, permute_76, permute_77, alias_19, permute_78, permute_79, permute_84, permute_85, div_23, permute_86, permute_87, permute_88, permute_89, div_24, permute_90, permute_91, permute_93, permute_94, alias_21, permute_95, permute_96, permute_101, permute_102, div_26, permute_103, permute_104, permute_105, permute_106, div_27, permute_107, permute_108, permute_110, permute_111, alias_23, permute_112, permute_113, permute_118, permute_119, div_29, permute_120, permute_121, permute_122, permute_123, div_30, permute_124, permute_125, permute_127, permute_128, alias_25, permute_129, permute_130, permute_135, permute_136, div_32, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14 = args
    args.clear()
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_85, (1, 512), (512, 1))
    assert_size_stride(view, (1, 512), (512, 1))
    assert_size_stride(view_1, (1, 512), (512, 1))
    assert_size_stride(getitem_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_4, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_8, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_2, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_2, (512, 3072), (3072, 1))
    assert_size_stride(tanh, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_14, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_8, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_21, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_23, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_6, (512, 3072), (3072, 1))
    assert_size_stride(tanh_1, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_12, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_34, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_18, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(tanh_2, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_16, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_47, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_26, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_14, (512, 3072), (3072, 1))
    assert_size_stride(tanh_3, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_53, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_20, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_60, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_62, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_34, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_18, (512, 3072), (3072, 1))
    assert_size_stride(tanh_4, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_66, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_24, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem_73, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(getitem_75, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_42, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(tanh_5, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(getitem_79, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_111, (512, 768), (768, 1))
    assert_size_stride(sub_20, (511, 50257), (50257, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_33, (50257, 768), (768, 1))
    assert_size_stride(div_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_35, (768, 3072), (1, 768))
    assert_size_stride(permute_36, (3072, 512), (1, 3072))
    assert_size_stride(permute_37, (3072, 768), (1, 3072))
    assert_size_stride(permute_38, (768, 512), (1, 768))
    assert_size_stride(div_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_39, (768, 768), (1, 768))
    assert_size_stride(permute_40, (768, 512), (1, 768))
    assert_size_stride(permute_42, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_43, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_15, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_44, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_45, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_50, (2304, 768), (1, 2304))
    assert_size_stride(permute_51, (768, 512), (1, 768))
    assert_size_stride(div_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_52, (768, 3072), (1, 768))
    assert_size_stride(permute_53, (3072, 512), (1, 3072))
    assert_size_stride(permute_54, (3072, 768), (1, 3072))
    assert_size_stride(permute_55, (768, 512), (1, 768))
    assert_size_stride(div_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_56, (768, 768), (1, 768))
    assert_size_stride(permute_57, (768, 512), (1, 768))
    assert_size_stride(permute_59, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_60, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_17, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_61, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_62, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_67, (2304, 768), (1, 2304))
    assert_size_stride(permute_68, (768, 512), (1, 768))
    assert_size_stride(div_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_69, (768, 3072), (1, 768))
    assert_size_stride(permute_70, (3072, 512), (1, 3072))
    assert_size_stride(permute_71, (3072, 768), (1, 3072))
    assert_size_stride(permute_72, (768, 512), (1, 768))
    assert_size_stride(div_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_73, (768, 768), (1, 768))
    assert_size_stride(permute_74, (768, 512), (1, 768))
    assert_size_stride(permute_76, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_77, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_19, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_78, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_79, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_84, (2304, 768), (1, 2304))
    assert_size_stride(permute_85, (768, 512), (1, 768))
    assert_size_stride(div_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_86, (768, 3072), (1, 768))
    assert_size_stride(permute_87, (3072, 512), (1, 3072))
    assert_size_stride(permute_88, (3072, 768), (1, 3072))
    assert_size_stride(permute_89, (768, 512), (1, 768))
    assert_size_stride(div_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_90, (768, 768), (1, 768))
    assert_size_stride(permute_91, (768, 512), (1, 768))
    assert_size_stride(permute_93, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_94, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_21, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_95, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_96, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_101, (2304, 768), (1, 2304))
    assert_size_stride(permute_102, (768, 512), (1, 768))
    assert_size_stride(div_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_103, (768, 3072), (1, 768))
    assert_size_stride(permute_104, (3072, 512), (1, 3072))
    assert_size_stride(permute_105, (3072, 768), (1, 3072))
    assert_size_stride(permute_106, (768, 512), (1, 768))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_107, (768, 768), (1, 768))
    assert_size_stride(permute_108, (768, 512), (1, 768))
    assert_size_stride(permute_110, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_111, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_23, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_112, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_113, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_118, (2304, 768), (1, 2304))
    assert_size_stride(permute_119, (768, 512), (1, 768))
    assert_size_stride(div_29, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_120, (768, 3072), (1, 768))
    assert_size_stride(permute_121, (3072, 512), (1, 3072))
    assert_size_stride(permute_122, (3072, 768), (1, 3072))
    assert_size_stride(permute_123, (768, 512), (1, 768))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_124, (768, 768), (1, 768))
    assert_size_stride(permute_125, (768, 512), (1, 768))
    assert_size_stride(permute_127, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_128, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(alias_25, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_129, (12, 64, 512), (64, 1, 2304))
    assert_size_stride(permute_130, (12, 512, 64), (64, 2304, 1))
    assert_size_stride(permute_135, (2304, 768), (1, 2304))
    assert_size_stride(permute_136, (768, 512), (1, 768))
    assert_size_stride(div_32, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 50257), (25731584, 50257, 1))
    assert_size_stride(tangents_3, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_4, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_5, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_6, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_7, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_8, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_9, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_10, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_11, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_12, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_13, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_14, (1, 12, 512, 64), (393216, 32768, 64, 1))
    buf0 = empty((511, 50257), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((511, 1), (1, 511), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_85.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((511, 1), (1, 511), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 512, 50257), device='cpu', dtype=torch.float32)
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_slice_backward_1(c_void_p(buf0.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_20.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del buf0
    del buf4
    del convert_element_type
    del primals_85
    del sub_20
    del tangents_1
    del tangents_2
    buf6 = empty((50257, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (50257, 512), (1, 50257), 0), view_111, out=buf6)
    del view_111
    buf7 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (512, 50257), (50257, 1), 0), permute_33, out=buf7)
    del buf5
    del permute_33
    buf8 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf10 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf11 = empty((768, ), device='cpu', dtype=torch.float32)
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_2(c_void_p(buf7.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del div_14
    del getitem_79
    del mul_48
    del primals_75
    buf14 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_35, out=buf14)
    del permute_35
    buf15 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_36, reinterpret_tensor(buf13, (512, 768), (768, 1), 0), out=buf15)
    del permute_36
    buf16 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf17 = reinterpret_tensor(buf14, (1, 512, 3072), (1572864, 3072, 1), 0); del buf14  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_3(c_void_p(buf17.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(tanh_5.data_ptr()), c_void_p(buf16.data_ptr()))
    del addmm_22
    del tanh_5
    buf18 = reinterpret_tensor(buf13, (512, 768), (768, 1), 0); del buf13  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0), permute_37, out=buf18)
    del permute_37
    buf19 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_38, reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0), out=buf19)
    del permute_38
    buf20 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf21 = buf9; del buf9  # reuse
    buf22 = buf8; del buf8  # reuse
    buf23 = empty((768, ), device='cpu', dtype=torch.float32)
    buf24 = empty((768, ), device='cpu', dtype=torch.float32)
    buf25 = buf10; del buf10  # reuse
    buf26 = reinterpret_tensor(buf7, (1, 512, 768), (393216, 768, 1), 0); del buf7  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_4(c_void_p(buf25.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(getitem_75.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()))
    del div_15
    del getitem_75
    del mul_42
    del primals_73
    buf27 = buf18; del buf18  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (512, 768), (768, 1), 0), permute_39, out=buf27)
    del permute_39
    buf28 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_40, reinterpret_tensor(buf26, (512, 768), (768, 1), 0), out=buf28)
    del permute_40
    buf29 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_5(c_void_p(buf26.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf26, (12, 512, 64), (32768, 64, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_42, reinterpret_tensor(buf27, (12, 512, 64), (64, 768, 1), 0), out=buf30)
    del permute_42
    buf31 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf27, (12, 512, 64), (64, 768, 1), 0), permute_43, out=buf31)
    del permute_43
    buf32 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf33 = reinterpret_tensor(buf31, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf31  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_6(c_void_p(buf33.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(slice_24.data_ptr()), c_void_p(buf32.data_ptr()))
    del alias_15
    del getitem_73
    del slice_24
    buf34 = reinterpret_tensor(buf27, (12, 64, 512), (32768, 512, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_44, reinterpret_tensor(buf33, (12, 512, 512), (262144, 512, 1), 0), out=buf34)
    del permute_44
    buf35 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf33, (12, 512, 512), (262144, 512, 1), 0), permute_45, out=buf35)
    del permute_45
    buf36 = empty((1, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_7(c_void_p(buf35.data_ptr()), c_void_p(tangents_13.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(tangents_14.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf36.data_ptr()))
    del tangents_13
    del tangents_14
    buf37 = reinterpret_tensor(buf35, (512, 768), (768, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (512, 2304), (2304, 1), 0), permute_50, out=buf37)
    del permute_50
    buf38 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_51, reinterpret_tensor(buf36, (512, 2304), (2304, 1), 0), out=buf38)
    del permute_51
    buf39 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf40 = buf22; del buf22  # reuse
    buf41 = buf21; del buf21  # reuse
    buf42 = empty((768, ), device='cpu', dtype=torch.float32)
    buf43 = empty((768, ), device='cpu', dtype=torch.float32)
    buf44 = buf25; del buf25  # reuse
    buf45 = reinterpret_tensor(buf34, (1, 512, 768), (393216, 768, 1), 0); del buf34  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_8(c_void_p(buf44.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(getitem_66.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()))
    del div_17
    del getitem_66
    del mul_40
    del primals_71
    buf46 = reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0); del buf17  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (512, 768), (768, 1), 0), permute_52, out=buf46)
    del permute_52
    buf47 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_53, reinterpret_tensor(buf45, (512, 768), (768, 1), 0), out=buf47)
    del permute_53
    buf48 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf49 = reinterpret_tensor(buf46, (1, 512, 3072), (1572864, 3072, 1), 0); del buf46  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_9(c_void_p(buf49.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(tanh_4.data_ptr()), c_void_p(buf48.data_ptr()))
    del addmm_18
    del tanh_4
    buf50 = reinterpret_tensor(buf45, (512, 768), (768, 1), 0); del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (512, 3072), (3072, 1), 0), permute_54, out=buf50)
    del permute_54
    buf51 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_55, reinterpret_tensor(buf49, (512, 3072), (3072, 1), 0), out=buf51)
    del permute_55
    buf52 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf53 = buf41; del buf41  # reuse
    buf54 = buf40; del buf40  # reuse
    buf55 = empty((768, ), device='cpu', dtype=torch.float32)
    buf56 = empty((768, ), device='cpu', dtype=torch.float32)
    buf57 = buf44; del buf44  # reuse
    buf58 = reinterpret_tensor(buf37, (1, 512, 768), (393216, 768, 1), 0); del buf37  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_10(c_void_p(buf57.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(mul_34.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(getitem_62.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()))
    del div_18
    del getitem_62
    del mul_34
    del primals_69
    buf59 = buf50; del buf50  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (512, 768), (768, 1), 0), permute_56, out=buf59)
    del permute_56
    buf60 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_57, reinterpret_tensor(buf58, (512, 768), (768, 1), 0), out=buf60)
    del permute_57
    buf61 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_11(c_void_p(buf58.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = reinterpret_tensor(buf58, (12, 512, 64), (32768, 64, 1), 0); del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_59, reinterpret_tensor(buf59, (12, 512, 64), (64, 768, 1), 0), out=buf62)
    del permute_59
    buf63 = reinterpret_tensor(buf33, (12, 512, 512), (262144, 512, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf59, (12, 512, 64), (64, 768, 1), 0), permute_60, out=buf63)
    del permute_60
    buf64 = buf32; del buf32  # reuse
    buf65 = reinterpret_tensor(buf63, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf63  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_12(c_void_p(buf65.data_ptr()), c_void_p(getitem_60.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(slice_20.data_ptr()), c_void_p(buf64.data_ptr()))
    del alias_17
    del getitem_60
    del slice_20
    buf66 = reinterpret_tensor(buf59, (12, 64, 512), (32768, 512, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_61, reinterpret_tensor(buf65, (12, 512, 512), (262144, 512, 1), 0), out=buf66)
    del permute_61
    buf67 = buf30; del buf30  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf65, (12, 512, 512), (262144, 512, 1), 0), permute_62, out=buf67)
    del permute_62
    buf68 = buf36; del buf36  # reuse
    cpp_fused_cat_13(c_void_p(buf67.data_ptr()), c_void_p(tangents_11.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(tangents_12.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf68.data_ptr()))
    del tangents_11
    del tangents_12
    buf69 = reinterpret_tensor(buf67, (512, 768), (768, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (512, 2304), (2304, 1), 0), permute_67, out=buf69)
    del permute_67
    buf70 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_68, reinterpret_tensor(buf68, (512, 2304), (2304, 1), 0), out=buf70)
    del permute_68
    buf71 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf72 = buf54; del buf54  # reuse
    buf73 = buf53; del buf53  # reuse
    buf74 = empty((768, ), device='cpu', dtype=torch.float32)
    buf75 = empty((768, ), device='cpu', dtype=torch.float32)
    buf76 = buf57; del buf57  # reuse
    buf77 = reinterpret_tensor(buf66, (1, 512, 768), (393216, 768, 1), 0); del buf66  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_14(c_void_p(buf76.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()))
    del div_20
    del getitem_53
    del mul_32
    del primals_67
    buf78 = reinterpret_tensor(buf49, (512, 3072), (3072, 1), 0); del buf49  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (512, 768), (768, 1), 0), permute_69, out=buf78)
    del permute_69
    buf79 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_70, reinterpret_tensor(buf77, (512, 768), (768, 1), 0), out=buf79)
    del permute_70
    buf80 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf81 = reinterpret_tensor(buf78, (1, 512, 3072), (1572864, 3072, 1), 0); del buf78  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_15(c_void_p(buf81.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(tanh_3.data_ptr()), c_void_p(buf80.data_ptr()))
    del addmm_14
    del tanh_3
    buf82 = reinterpret_tensor(buf77, (512, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf81, (512, 3072), (3072, 1), 0), permute_71, out=buf82)
    del permute_71
    buf83 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_72, reinterpret_tensor(buf81, (512, 3072), (3072, 1), 0), out=buf83)
    del permute_72
    buf84 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf85 = buf73; del buf73  # reuse
    buf86 = buf72; del buf72  # reuse
    buf87 = empty((768, ), device='cpu', dtype=torch.float32)
    buf88 = empty((768, ), device='cpu', dtype=torch.float32)
    buf89 = buf76; del buf76  # reuse
    buf90 = reinterpret_tensor(buf69, (1, 512, 768), (393216, 768, 1), 0); del buf69  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_16(c_void_p(buf89.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(mul_26.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    del div_21
    del getitem_49
    del mul_26
    del primals_65
    buf91 = buf82; del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (512, 768), (768, 1), 0), permute_73, out=buf91)
    del permute_73
    buf92 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_74, reinterpret_tensor(buf90, (512, 768), (768, 1), 0), out=buf92)
    del permute_74
    buf93 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_17(c_void_p(buf90.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = reinterpret_tensor(buf90, (12, 512, 64), (32768, 64, 1), 0); del buf90  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_76, reinterpret_tensor(buf91, (12, 512, 64), (64, 768, 1), 0), out=buf94)
    del permute_76
    buf95 = reinterpret_tensor(buf65, (12, 512, 512), (262144, 512, 1), 0); del buf65  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf91, (12, 512, 64), (64, 768, 1), 0), permute_77, out=buf95)
    del permute_77
    buf96 = buf64; del buf64  # reuse
    buf97 = reinterpret_tensor(buf95, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf95  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_18(c_void_p(buf97.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(alias_19.data_ptr()), c_void_p(slice_16.data_ptr()), c_void_p(buf96.data_ptr()))
    del alias_19
    del getitem_47
    del slice_16
    buf98 = reinterpret_tensor(buf91, (12, 64, 512), (32768, 512, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_78, reinterpret_tensor(buf97, (12, 512, 512), (262144, 512, 1), 0), out=buf98)
    del permute_78
    buf99 = buf62; del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf97, (12, 512, 512), (262144, 512, 1), 0), permute_79, out=buf99)
    del permute_79
    buf100 = buf68; del buf68  # reuse
    cpp_fused_cat_19(c_void_p(buf99.data_ptr()), c_void_p(tangents_9.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(tangents_10.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf100.data_ptr()))
    del tangents_10
    del tangents_9
    buf101 = reinterpret_tensor(buf99, (512, 768), (768, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (512, 2304), (2304, 1), 0), permute_84, out=buf101)
    del permute_84
    buf102 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_85, reinterpret_tensor(buf100, (512, 2304), (2304, 1), 0), out=buf102)
    del permute_85
    buf103 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf104 = buf86; del buf86  # reuse
    buf105 = buf85; del buf85  # reuse
    buf106 = empty((768, ), device='cpu', dtype=torch.float32)
    buf107 = empty((768, ), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf101, (1, 512, 768), (393216, 768, 1), 0); del buf101  # reuse
    buf109 = reinterpret_tensor(buf98, (1, 512, 768), (393216, 768, 1), 0); del buf98  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_20(c_void_p(buf108.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(getitem_40.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    del div_23
    del getitem_40
    del mul_24
    del primals_63
    buf110 = reinterpret_tensor(buf81, (512, 3072), (3072, 1), 0); del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (512, 768), (768, 1), 0), permute_86, out=buf110)
    del permute_86
    buf111 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_87, reinterpret_tensor(buf109, (512, 768), (768, 1), 0), out=buf111)
    del permute_87
    buf112 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf113 = reinterpret_tensor(buf110, (1, 512, 3072), (1572864, 3072, 1), 0); del buf110  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_21(c_void_p(buf113.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(tanh_2.data_ptr()), c_void_p(buf112.data_ptr()))
    del addmm_10
    del tanh_2
    buf114 = reinterpret_tensor(buf109, (512, 768), (768, 1), 0); del buf109  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf113, (512, 3072), (3072, 1), 0), permute_88, out=buf114)
    del permute_88
    buf115 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_89, reinterpret_tensor(buf113, (512, 3072), (3072, 1), 0), out=buf115)
    del permute_89
    buf116 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf117 = buf105; del buf105  # reuse
    buf118 = buf104; del buf104  # reuse
    buf119 = empty((768, ), device='cpu', dtype=torch.float32)
    buf120 = empty((768, ), device='cpu', dtype=torch.float32)
    buf121 = buf108; del buf108  # reuse
    buf122 = buf89; del buf89  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_22(c_void_p(buf121.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(mul_18.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(getitem_36.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()))
    del div_24
    del getitem_36
    del mul_18
    del primals_61
    buf123 = buf114; del buf114  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (512, 768), (768, 1), 0), permute_90, out=buf123)
    del permute_90
    buf124 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_91, reinterpret_tensor(buf122, (512, 768), (768, 1), 0), out=buf124)
    del permute_91
    buf125 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_23(c_void_p(buf122.data_ptr()), c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf122, (12, 512, 64), (32768, 64, 1), 0); del buf122  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_93, reinterpret_tensor(buf123, (12, 512, 64), (64, 768, 1), 0), out=buf126)
    del permute_93
    buf127 = reinterpret_tensor(buf97, (12, 512, 512), (262144, 512, 1), 0); del buf97  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf123, (12, 512, 64), (64, 768, 1), 0), permute_94, out=buf127)
    del permute_94
    buf128 = buf96; del buf96  # reuse
    buf129 = reinterpret_tensor(buf127, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf127  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_24(c_void_p(buf129.data_ptr()), c_void_p(getitem_34.data_ptr()), c_void_p(alias_21.data_ptr()), c_void_p(slice_12.data_ptr()), c_void_p(buf128.data_ptr()))
    del alias_21
    del getitem_34
    del slice_12
    buf130 = reinterpret_tensor(buf123, (12, 64, 512), (32768, 512, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_95, reinterpret_tensor(buf129, (12, 512, 512), (262144, 512, 1), 0), out=buf130)
    del permute_95
    buf131 = buf94; del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf129, (12, 512, 512), (262144, 512, 1), 0), permute_96, out=buf131)
    del permute_96
    buf132 = buf100; del buf100  # reuse
    cpp_fused_cat_25(c_void_p(buf131.data_ptr()), c_void_p(tangents_7.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(tangents_8.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf132.data_ptr()))
    del tangents_7
    del tangents_8
    buf133 = reinterpret_tensor(buf131, (512, 768), (768, 1), 0); del buf131  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (512, 2304), (2304, 1), 0), permute_101, out=buf133)
    del permute_101
    buf134 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_102, reinterpret_tensor(buf132, (512, 2304), (2304, 1), 0), out=buf134)
    del permute_102
    buf135 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf136 = buf118; del buf118  # reuse
    buf137 = buf117; del buf117  # reuse
    buf138 = empty((768, ), device='cpu', dtype=torch.float32)
    buf139 = empty((768, ), device='cpu', dtype=torch.float32)
    buf140 = buf121; del buf121  # reuse
    buf141 = reinterpret_tensor(buf130, (1, 512, 768), (393216, 768, 1), 0); del buf130  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_26(c_void_p(buf140.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()))
    del div_26
    del getitem_27
    del mul_16
    del primals_59
    buf142 = reinterpret_tensor(buf113, (512, 3072), (3072, 1), 0); del buf113  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf141, (512, 768), (768, 1), 0), permute_103, out=buf142)
    del permute_103
    buf143 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_104, reinterpret_tensor(buf141, (512, 768), (768, 1), 0), out=buf143)
    del permute_104
    buf144 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf145 = reinterpret_tensor(buf142, (1, 512, 3072), (1572864, 3072, 1), 0); del buf142  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_27(c_void_p(buf145.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(tanh_1.data_ptr()), c_void_p(buf144.data_ptr()))
    del addmm_6
    del tanh_1
    buf146 = reinterpret_tensor(buf141, (512, 768), (768, 1), 0); del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (512, 3072), (3072, 1), 0), permute_105, out=buf146)
    del permute_105
    buf147 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_106, reinterpret_tensor(buf145, (512, 3072), (3072, 1), 0), out=buf147)
    del permute_106
    buf148 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf149 = buf137; del buf137  # reuse
    buf150 = buf136; del buf136  # reuse
    buf151 = empty((768, ), device='cpu', dtype=torch.float32)
    buf152 = empty((768, ), device='cpu', dtype=torch.float32)
    buf153 = buf140; del buf140  # reuse
    buf154 = reinterpret_tensor(buf133, (1, 512, 768), (393216, 768, 1), 0); del buf133  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_28(c_void_p(buf153.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(getitem_23.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()))
    del div_27
    del getitem_23
    del mul_10
    del primals_57
    buf155 = buf146; del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (512, 768), (768, 1), 0), permute_107, out=buf155)
    del permute_107
    buf156 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_108, reinterpret_tensor(buf154, (512, 768), (768, 1), 0), out=buf156)
    del permute_108
    buf157 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_29(c_void_p(buf154.data_ptr()), c_void_p(buf157.data_ptr()))
    buf158 = reinterpret_tensor(buf154, (12, 512, 64), (32768, 64, 1), 0); del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_110, reinterpret_tensor(buf155, (12, 512, 64), (64, 768, 1), 0), out=buf158)
    del permute_110
    buf159 = reinterpret_tensor(buf129, (12, 512, 512), (262144, 512, 1), 0); del buf129  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf155, (12, 512, 64), (64, 768, 1), 0), permute_111, out=buf159)
    del permute_111
    buf160 = buf128; del buf128  # reuse
    buf161 = reinterpret_tensor(buf159, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf159  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_30(c_void_p(buf161.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(alias_23.data_ptr()), c_void_p(slice_8.data_ptr()), c_void_p(buf160.data_ptr()))
    del alias_23
    del getitem_21
    del slice_8
    buf162 = reinterpret_tensor(buf155, (12, 64, 512), (32768, 512, 1), 0); del buf155  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_112, reinterpret_tensor(buf161, (12, 512, 512), (262144, 512, 1), 0), out=buf162)
    del permute_112
    buf163 = buf126; del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf161, (12, 512, 512), (262144, 512, 1), 0), permute_113, out=buf163)
    del permute_113
    buf164 = buf132; del buf132  # reuse
    cpp_fused_cat_31(c_void_p(buf163.data_ptr()), c_void_p(tangents_5.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(tangents_6.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf164.data_ptr()))
    del tangents_5
    del tangents_6
    buf165 = reinterpret_tensor(buf163, (512, 768), (768, 1), 0); del buf163  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf164, (512, 2304), (2304, 1), 0), permute_118, out=buf165)
    del permute_118
    buf166 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_119, reinterpret_tensor(buf164, (512, 2304), (2304, 1), 0), out=buf166)
    del permute_119
    buf167 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf168 = buf150; del buf150  # reuse
    buf169 = buf149; del buf149  # reuse
    buf170 = empty((768, ), device='cpu', dtype=torch.float32)
    buf171 = empty((768, ), device='cpu', dtype=torch.float32)
    buf172 = buf153; del buf153  # reuse
    buf173 = reinterpret_tensor(buf162, (1, 512, 768), (393216, 768, 1), 0); del buf162  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_32(c_void_p(buf172.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(getitem_14.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()))
    del div_29
    del getitem_14
    del mul_8
    del primals_55
    buf174 = reinterpret_tensor(buf145, (512, 3072), (3072, 1), 0); del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (512, 768), (768, 1), 0), permute_120, out=buf174)
    del permute_120
    buf175 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_121, reinterpret_tensor(buf173, (512, 768), (768, 1), 0), out=buf175)
    del permute_121
    buf176 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf177 = reinterpret_tensor(buf174, (1, 512, 3072), (1572864, 3072, 1), 0); del buf174  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_33(c_void_p(buf177.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(buf176.data_ptr()))
    del addmm_2
    del tanh
    buf178 = reinterpret_tensor(buf173, (512, 768), (768, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (512, 3072), (3072, 1), 0), permute_122, out=buf178)
    del permute_122
    buf179 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_123, reinterpret_tensor(buf177, (512, 3072), (3072, 1), 0), out=buf179)
    del permute_123
    buf180 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf181 = buf169; del buf169  # reuse
    buf182 = buf168; del buf168  # reuse
    buf183 = empty((768, ), device='cpu', dtype=torch.float32)
    buf184 = empty((768, ), device='cpu', dtype=torch.float32)
    buf185 = buf172; del buf172  # reuse
    buf186 = reinterpret_tensor(buf165, (1, 512, 768), (393216, 768, 1), 0); del buf165  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_34(c_void_p(buf185.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(getitem_10.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()))
    del buf177
    del div_30
    del getitem_10
    del mul_2
    del primals_53
    buf187 = buf178; del buf178  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (512, 768), (768, 1), 0), permute_124, out=buf187)
    del permute_124
    buf188 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_125, reinterpret_tensor(buf186, (512, 768), (768, 1), 0), out=buf188)
    del permute_125
    buf189 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_35(c_void_p(buf186.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = reinterpret_tensor(buf186, (12, 512, 64), (32768, 64, 1), 0); del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_127, reinterpret_tensor(buf187, (12, 512, 64), (64, 768, 1), 0), out=buf190)
    del permute_127
    buf191 = reinterpret_tensor(buf161, (12, 512, 512), (262144, 512, 1), 0); del buf161  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf187, (12, 512, 64), (64, 768, 1), 0), permute_128, out=buf191)
    del permute_128
    buf192 = buf160; del buf160  # reuse
    buf193 = reinterpret_tensor(buf191, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf191  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_36(c_void_p(buf193.data_ptr()), c_void_p(getitem_8.data_ptr()), c_void_p(alias_25.data_ptr()), c_void_p(slice_4.data_ptr()), c_void_p(buf192.data_ptr()))
    del alias_25
    del buf192
    del getitem_8
    del slice_4
    buf194 = reinterpret_tensor(buf187, (12, 64, 512), (32768, 512, 1), 0); del buf187  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_129, reinterpret_tensor(buf193, (12, 512, 512), (262144, 512, 1), 0), out=buf194)
    del permute_129
    buf195 = buf158; del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf193, (12, 512, 512), (262144, 512, 1), 0), permute_130, out=buf195)
    del buf193
    del permute_130
    buf196 = buf164; del buf164  # reuse
    cpp_fused_cat_37(c_void_p(buf195.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(tangents_4.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf196.data_ptr()))
    del buf190
    del tangents_3
    del tangents_4
    buf197 = reinterpret_tensor(buf195, (512, 768), (768, 1), 0); del buf195  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (512, 2304), (2304, 1), 0), permute_135, out=buf197)
    del permute_135
    buf198 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_136, reinterpret_tensor(buf196, (512, 2304), (2304, 1), 0), out=buf198)
    del permute_136
    buf199 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf200 = buf182; del buf182  # reuse
    buf201 = buf181; del buf181  # reuse
    buf202 = empty((768, ), device='cpu', dtype=torch.float32)
    buf203 = empty((768, ), device='cpu', dtype=torch.float32)
    buf204 = buf185; del buf185  # reuse
    buf210 = reinterpret_tensor(buf194, (1, 512, 768), (393216, 768, 1), 0); del buf194  # reuse
    buf205 = empty((1024, 768), device='cpu', dtype=torch.float32)
    buf206 = buf204; del buf204  # reuse
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_38(c_void_p(buf206.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(view.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf205.data_ptr()))
    del buf196
    del buf197
    del buf200
    del buf201
    del div_32
    del getitem_1
    del mul
    del primals_51
    aten.index_put_(buf205, [view_1], buf206, True)
    del buf206
    del view_1
    buf209 = empty((50257, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_39(c_void_p(buf209.data_ptr()))
    aten.index_put_(buf209, [view], buf210, True)
    del buf210
    del view
    return (reinterpret_tensor(buf199, (2304, ), (1, ), 0), buf198, reinterpret_tensor(buf189, (768, ), (1, ), 0), buf188, reinterpret_tensor(buf180, (3072, ), (1, ), 0), buf179, reinterpret_tensor(buf176, (768, ), (1, ), 0), buf175, reinterpret_tensor(buf167, (2304, ), (1, ), 0), buf166, reinterpret_tensor(buf157, (768, ), (1, ), 0), buf156, reinterpret_tensor(buf148, (3072, ), (1, ), 0), buf147, reinterpret_tensor(buf144, (768, ), (1, ), 0), buf143, reinterpret_tensor(buf135, (2304, ), (1, ), 0), buf134, reinterpret_tensor(buf125, (768, ), (1, ), 0), buf124, reinterpret_tensor(buf116, (3072, ), (1, ), 0), buf115, reinterpret_tensor(buf112, (768, ), (1, ), 0), buf111, reinterpret_tensor(buf103, (2304, ), (1, ), 0), buf102, reinterpret_tensor(buf93, (768, ), (1, ), 0), buf92, reinterpret_tensor(buf84, (3072, ), (1, ), 0), buf83, reinterpret_tensor(buf80, (768, ), (1, ), 0), buf79, reinterpret_tensor(buf71, (2304, ), (1, ), 0), buf70, reinterpret_tensor(buf61, (768, ), (1, ), 0), buf60, reinterpret_tensor(buf52, (3072, ), (1, ), 0), buf51, reinterpret_tensor(buf48, (768, ), (1, ), 0), buf47, reinterpret_tensor(buf39, (2304, ), (1, ), 0), buf38, reinterpret_tensor(buf29, (768, ), (1, ), 0), buf28, reinterpret_tensor(buf20, (3072, ), (1, ), 0), buf19, reinterpret_tensor(buf16, (768, ), (1, ), 0), buf15, buf209, buf205, buf202, buf203, buf183, buf184, buf170, buf171, buf151, buf152, buf138, buf139, buf119, buf120, buf106, buf107, buf87, buf88, buf74, buf75, buf55, buf56, buf42, buf43, buf23, buf24, buf11, buf12, reinterpret_tensor(buf6, (50257, 768), (768, 1), 0), None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    view = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    view_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    getitem_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_4 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_8 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    getitem_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_2 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_14 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_8 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_21 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    getitem_23 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_12 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_34 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    getitem_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_18 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_16 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_47 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    getitem_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_26 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_20 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_60 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    getitem_62 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_34 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_66 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_24 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_73 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    getitem_75 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_42 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_111 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_20 = rand_strided((511, 50257), (50257, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_33 = rand_strided((50257, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_35 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_36 = rand_strided((3072, 512), (1, 3072), device='cpu', dtype=torch.float32)
    permute_37 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_38 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_39 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_40 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    permute_42 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_43 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_44 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_45 = rand_strided((12, 512, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_50 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_51 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_52 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_53 = rand_strided((3072, 512), (1, 3072), device='cpu', dtype=torch.float32)
    permute_54 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_55 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_56 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_57 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    permute_59 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_60 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_61 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_62 = rand_strided((12, 512, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_68 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_69 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_70 = rand_strided((3072, 512), (1, 3072), device='cpu', dtype=torch.float32)
    permute_71 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_72 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_73 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_74 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    permute_76 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_77 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_19 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_78 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_79 = rand_strided((12, 512, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_84 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_85 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_86 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_87 = rand_strided((3072, 512), (1, 3072), device='cpu', dtype=torch.float32)
    permute_88 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_89 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_90 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_91 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    permute_93 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_94 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_21 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_95 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_96 = rand_strided((12, 512, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_101 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_102 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_103 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((3072, 512), (1, 3072), device='cpu', dtype=torch.float32)
    permute_105 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_106 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_107 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_108 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    permute_110 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_111 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_23 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_112 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_113 = rand_strided((12, 512, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_119 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_120 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_121 = rand_strided((3072, 512), (1, 3072), device='cpu', dtype=torch.float32)
    permute_122 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_124 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_125 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    permute_127 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_128 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_25 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_129 = rand_strided((12, 64, 512), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_130 = rand_strided((12, 512, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_135 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((768, 512), (1, 768), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 50257), (25731584, 50257, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_85, view, view_1, getitem_1, mul, slice_4, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, slice_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, slice_12, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, slice_16, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, slice_20, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, slice_24, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, view_111, sub_20, convert_element_type, permute_33, div_14, permute_35, permute_36, permute_37, permute_38, div_15, permute_39, permute_40, permute_42, permute_43, alias_15, permute_44, permute_45, permute_50, permute_51, div_17, permute_52, permute_53, permute_54, permute_55, div_18, permute_56, permute_57, permute_59, permute_60, alias_17, permute_61, permute_62, permute_67, permute_68, div_20, permute_69, permute_70, permute_71, permute_72, div_21, permute_73, permute_74, permute_76, permute_77, alias_19, permute_78, permute_79, permute_84, permute_85, div_23, permute_86, permute_87, permute_88, permute_89, div_24, permute_90, permute_91, permute_93, permute_94, alias_21, permute_95, permute_96, permute_101, permute_102, div_26, permute_103, permute_104, permute_105, permute_106, div_27, permute_107, permute_108, permute_110, permute_111, alias_23, permute_112, permute_113, permute_118, permute_119, div_29, permute_120, permute_121, permute_122, permute_123, div_30, permute_124, permute_125, permute_127, permute_128, alias_25, permute_129, permute_130, permute_135, permute_136, div_32, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistillGPT2', benchmark_compiled_module)
