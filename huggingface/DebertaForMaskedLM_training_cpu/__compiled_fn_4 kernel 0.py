
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25735680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
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


cpp_fused__log_softmax_backward_data_add_masked_fill_nll_loss_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
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
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (50265L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (50265L*x0)));
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<int>(-100);
                    auto tmp4 = tmp2 != tmp3;
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = tmp4 ? tmp7 : tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp1 * tmp10;
                    auto tmp13 = tmp12.exp();
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp11 - tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x1 + (50265L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp11 = in_ptr5[static_cast<long>(x1 + (50265L*x0))];
                    auto tmp13 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<long>(-100);
                    auto tmp4 = tmp2 != tmp3;
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = tmp4 ? tmp7 : tmp8;
                    auto tmp10 = decltype(tmp1)(tmp1 * tmp9);
                    auto tmp12 = std::exp(tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp10)(tmp10 - tmp14);
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    in_out_ptr0[static_cast<long>(x1 + (50265L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_native_layer_norm_backward_sum_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50264L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50265L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(50264L); x0<static_cast<long>(50265L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (50265L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    auto tmp18 = static_cast<float>(0.7071067811865476);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 * tmp19;
                    auto tmp21 = tmp20.erf();
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = at::vec::Vectorized<float>(tmp22);
                    auto tmp24 = tmp21 + tmp23;
                    auto tmp25 = static_cast<float>(0.5);
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp24 * tmp26;
                    auto tmp28 = tmp17 * tmp17;
                    auto tmp29 = static_cast<float>(-0.5);
                    auto tmp30 = at::vec::Vectorized<float>(tmp29);
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp31.exp();
                    auto tmp33 = static_cast<float>(0.3989422804014327);
                    auto tmp34 = at::vec::Vectorized<float>(tmp33);
                    auto tmp35 = tmp32 * tmp34;
                    auto tmp36 = tmp17 * tmp35;
                    auto tmp37 = tmp27 + tmp36;
                    auto tmp38 = tmp16 * tmp37;
                    tmp38.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = in_ptr3[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp3 = tmp2.neg();
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp7 / tmp6;
                        auto tmp9 = tmp3 * tmp8;
                        auto tmp10 = tmp2 / tmp6;
                        auto tmp11 = tmp10.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = in_ptr3[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp18 = out_ptr4[static_cast<long>(x0)];
                    auto tmp19 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 / tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = static_cast<float>(768.0);
                    auto tmp11 = tmp9 / tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp7);
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = at::vec::Vectorized<float>(tmp11);
                    auto tmp16 = tmp15 * tmp14;
                    auto tmp17 = tmp5 + tmp16;
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = tmp20 / tmp10;
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 + tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_4 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_5 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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


cpp_fused__softmax_backward_data_masked_fill_mul_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_10 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_11 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_12 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_16 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_17 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_18 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_22 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_23 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_24 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_28 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_29 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_30 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_34 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_35 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_36 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_40 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_41 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_42 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_46 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_47 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_48 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_52 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_53 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_54 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_58 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_59 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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


cpp_fused__softmax_backward_data_masked_fill_mul_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_64 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_65 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_66 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_70 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_71 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_72 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_masked_fill_mul_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const long* in_ptr6,
                       const long* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    auto out_ptr2 = in_out_ptr0;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp11 = in_ptr4[static_cast<long>(x1)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                        auto tmp7 = static_cast<float>(1.1111111111111112);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp12 = in_ptr4[static_cast<long>(x0)];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                        auto tmp7 = static_cast<float>(1.1111111111111112);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 / tmp13;
                        auto tmp15 = tmp11.neg();
                        auto tmp17 = tmp16 / tmp13;
                        auto tmp18 = tmp17 / tmp13;
                        auto tmp19 = tmp15 * tmp18;
                        tmp14.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp19;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = tmp0.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr4[static_cast<long>(x0)];
                    auto tmp15 = out_ptr5[static_cast<long>(x0)];
                    auto tmp20 = in_ptr6[static_cast<long>(x0)];
                    auto tmp27 = in_ptr7[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(2.0);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = tmp1 / tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp3);
                    auto tmp10 = tmp8 * tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp7);
                    auto tmp12 = tmp11 * tmp10;
                    auto tmp13 = tmp0 + tmp12;
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = tmp16 / tmp6;
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp13 + tmp18;
                    auto tmp21 = static_cast<int>(-1);
                    auto tmp22 = tmp20 == tmp21;
                    auto tmp23 = static_cast<float>(0.0);
                    auto tmp24 = to_float_mask(tmp22);
                    auto tmp25 = at::vec::Vectorized<float>(tmp23);
                    auto tmp26 = decltype(tmp25)::blendv(tmp19, tmp25, tmp24);
                    auto tmp28 = static_cast<int>(0);
                    auto tmp29 = tmp27 == tmp28;
                    auto tmp30 = to_float_mask(tmp29);
                    auto tmp31 = decltype(tmp25)::blendv(tmp19, tmp25, tmp30);
                    tmp26.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    tmp31.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr8 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(38603520L); x0+=static_cast<long>(8L))
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
    primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_163, primals_168, primals_169, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, addmm_36, mul_115, view_218, sub_101, convert_element_type_49, permute_147, div_51, permute_151, permute_155, permute_159, permute_163, permute_168, permute_169, alias_43, permute_170, permute_171, permute_178, permute_180, permute_184, permute_188, permute_193, permute_194, alias_48, permute_195, permute_196, permute_203, permute_205, permute_209, permute_213, permute_218, permute_219, alias_53, permute_220, permute_221, permute_228, permute_230, permute_234, permute_238, permute_243, permute_244, alias_58, permute_245, permute_246, permute_253, permute_255, permute_259, permute_263, permute_268, permute_269, alias_63, permute_270, permute_271, permute_278, permute_280, permute_284, permute_288, permute_293, permute_294, alias_68, permute_295, permute_296, permute_303, permute_305, permute_309, permute_313, permute_318, permute_319, alias_73, permute_320, permute_321, permute_328, permute_330, permute_334, permute_338, permute_343, permute_344, alias_78, permute_345, permute_346, permute_353, permute_355, permute_359, permute_363, permute_368, permute_369, alias_83, permute_370, permute_371, permute_378, permute_380, permute_384, permute_388, permute_393, permute_394, alias_88, permute_395, permute_396, permute_403, permute_405, permute_409, permute_413, permute_418, permute_419, alias_93, permute_420, permute_421, permute_428, permute_430, permute_434, permute_438, permute_443, permute_444, alias_98, permute_445, permute_446, permute_453, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_168, (1, 512), (512, 1))
    assert_size_stride(primals_169, (1, 512), (512, 1))
    assert_size_stride(slice_1, (1, 512), (512, 1))
    assert_size_stride(sub, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt, (1, 512, 1), (512, 1, 1))
    assert_size_stride(convert_element_type, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_2, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_12, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_6, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_2, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_14, (512, 768), (768, 1))
    assert_size_stride(addmm_1, (512, 3072), (3072, 1))
    assert_size_stride(view_16, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_4, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_9, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_3, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_6, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_14, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_5, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_32, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_34, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_6, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_36, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_10, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_48, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_22, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_8, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_50, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 3072), (3072, 1))
    assert_size_stride(view_52, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_12, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_9, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_54, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_14, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_15, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_30, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_11, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_68, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_70, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_33, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_12, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_72, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_18, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_19, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_38, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_86, (512, 768), (768, 1))
    assert_size_stride(addmm_13, (512, 3072), (3072, 1))
    assert_size_stride(view_88, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_20, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_90, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_22, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_102, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_23, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_46, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_106, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_108, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_26, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_120, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_54, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_122, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(view_124, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_28, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_30, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_138, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_62, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_140, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_142, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_65, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_144, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_34, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_156, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_35, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_70, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_158, (512, 768), (768, 1))
    assert_size_stride(addmm_25, (512, 3072), (3072, 1))
    assert_size_stride(view_160, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_162, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_38, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_174, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_39, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_78, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_29, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_178, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_180, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_42, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_86, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_32, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_31, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_44, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_89, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_46, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_210, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_94, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_35, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_212, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_214, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_36, (512, 768), (768, 1))
    assert_size_stride(mul_115, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_218, (512, 768), (768, 1))
    assert_size_stride(sub_101, (512, 50265), (50265, 1))
    assert_size_stride(convert_element_type_49, (), ())
    assert_size_stride(permute_147, (50265, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_151, (768, 768), (768, 1))
    assert_size_stride(permute_155, (768, 3072), (3072, 1))
    assert_size_stride(permute_159, (3072, 768), (768, 1))
    assert_size_stride(permute_163, (768, 768), (768, 1))
    assert_size_stride(permute_168, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_169, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_43, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_170, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_171, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_178, (2304, 768), (768, 1))
    assert_size_stride(permute_180, (768, 3072), (3072, 1))
    assert_size_stride(permute_184, (3072, 768), (768, 1))
    assert_size_stride(permute_188, (768, 768), (768, 1))
    assert_size_stride(permute_193, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_194, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_48, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_195, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_196, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_203, (2304, 768), (768, 1))
    assert_size_stride(permute_205, (768, 3072), (3072, 1))
    assert_size_stride(permute_209, (3072, 768), (768, 1))
    assert_size_stride(permute_213, (768, 768), (768, 1))
    assert_size_stride(permute_218, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_219, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_53, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_220, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_221, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_228, (2304, 768), (768, 1))
    assert_size_stride(permute_230, (768, 3072), (3072, 1))
    assert_size_stride(permute_234, (3072, 768), (768, 1))
    assert_size_stride(permute_238, (768, 768), (768, 1))
    assert_size_stride(permute_243, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_244, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_58, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_245, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_246, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_253, (2304, 768), (768, 1))
    assert_size_stride(permute_255, (768, 3072), (3072, 1))
    assert_size_stride(permute_259, (3072, 768), (768, 1))
    assert_size_stride(permute_263, (768, 768), (768, 1))
    assert_size_stride(permute_268, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_269, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_63, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_270, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_271, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_278, (2304, 768), (768, 1))
    assert_size_stride(permute_280, (768, 3072), (3072, 1))
    assert_size_stride(permute_284, (3072, 768), (768, 1))
    assert_size_stride(permute_288, (768, 768), (768, 1))
    assert_size_stride(permute_293, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_294, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_68, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_295, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_296, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_303, (2304, 768), (768, 1))
    assert_size_stride(permute_305, (768, 3072), (3072, 1))
    assert_size_stride(permute_309, (3072, 768), (768, 1))
    assert_size_stride(permute_313, (768, 768), (768, 1))
    assert_size_stride(permute_318, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_319, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_73, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_320, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_321, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_328, (2304, 768), (768, 1))
    assert_size_stride(permute_330, (768, 3072), (3072, 1))
    assert_size_stride(permute_334, (3072, 768), (768, 1))
    assert_size_stride(permute_338, (768, 768), (768, 1))
    assert_size_stride(permute_343, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_344, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_78, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_345, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_346, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_353, (2304, 768), (768, 1))
    assert_size_stride(permute_355, (768, 3072), (3072, 1))
    assert_size_stride(permute_359, (3072, 768), (768, 1))
    assert_size_stride(permute_363, (768, 768), (768, 1))
    assert_size_stride(permute_368, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_369, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_83, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_370, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_371, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_378, (2304, 768), (768, 1))
    assert_size_stride(permute_380, (768, 3072), (3072, 1))
    assert_size_stride(permute_384, (3072, 768), (768, 1))
    assert_size_stride(permute_388, (768, 768), (768, 1))
    assert_size_stride(permute_393, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_394, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_88, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_395, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_396, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_403, (2304, 768), (768, 1))
    assert_size_stride(permute_405, (768, 3072), (3072, 1))
    assert_size_stride(permute_409, (3072, 768), (768, 1))
    assert_size_stride(permute_413, (768, 768), (768, 1))
    assert_size_stride(permute_418, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_419, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_93, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_420, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_421, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_428, (2304, 768), (768, 1))
    assert_size_stride(permute_430, (768, 3072), (3072, 1))
    assert_size_stride(permute_434, (3072, 768), (768, 1))
    assert_size_stride(permute_438, (768, 768), (768, 1))
    assert_size_stride(permute_443, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_444, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_98, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_445, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_446, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_453, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 50265), (25735680, 50265, 1))
    buf0 = empty((512, 50265), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_169.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((512, 50265), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 512, 50265), (25735680, 50265, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_masked_fill_nll_loss_backward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type_49.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_101.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del convert_element_type_49
    del primals_169
    del sub_101
    del tangents_1
    del tangents_2
    buf6 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (512, 50265), (50265, 1), 0), permute_147, out=buf6)
    del permute_147
    buf7 = empty((50265, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (50265, 512), (1, 50265), 0), view_218, out=buf7)
    del view_218
    buf8 = empty((1, 50265), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf4, (1, 512, 1), (512, 1, 512), 0); del buf4  # reuse
    buf10 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf11 = empty((768, ), device='cpu', dtype=torch.float32)
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf6, (1, 512, 768), (393216, 768, 1), 0); del buf6  # reuse
    cpp_fused_gelu_gelu_backward_native_layer_norm_backward_sum_2(c_void_p(buf13.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(mul_115.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(addmm_36.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del addmm_36
    del buf5
    del div_51
    del mul_115
    del primals_163
    buf14 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_151, out=buf14)
    del permute_151
    buf15 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (768, 512), (1, 768), 0), view_216, out=buf15)
    del view_216
    buf16 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf17 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf18 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf19 = buf9; del buf9  # reuse
    buf20 = buf10; del buf10  # reuse
    buf21 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf22 = reinterpret_tensor(buf14, (1, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
    buf23 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_3(c_void_p(buf22.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(sub_97.data_ptr()), c_void_p(sqrt_36.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(convert_element_type_48.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del convert_element_type_48
    del primals_73
    del sqrt_36
    del sub_97
    buf24 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (512, 768), (768, 1), 0), permute_155, out=buf24)
    del permute_155
    buf25 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (768, 512), (1, 768), 0), view_214, out=buf25)
    del view_214
    buf26 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf24, (1, 512, 3072), (1572864, 3072, 1), 0); del buf24  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf27.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf26.data_ptr()))
    del addmm_34
    buf28 = reinterpret_tensor(buf23, (512, 768), (768, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0), permute_159, out=buf28)
    del permute_159
    buf29 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (3072, 512), (1, 3072), 0), view_212, out=buf29)
    del view_212
    buf30 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf31 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf32 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf33 = buf21; del buf21  # reuse
    buf34 = buf20; del buf20  # reuse
    buf35 = buf19; del buf19  # reuse
    buf36 = buf22; del buf22  # reuse
    buf37 = buf13; del buf13  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_5(c_void_p(buf36.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(sub_94.data_ptr()), c_void_p(sqrt_35.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(convert_element_type_47.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()))
    del convert_element_type_47
    del primals_71
    del sqrt_35
    del sub_94
    buf38 = buf28; del buf28  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (512, 768), (768, 1), 0), permute_163, out=buf38)
    del permute_163
    buf39 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (768, 512), (1, 768), 0), view_210, out=buf39)
    del view_210
    buf40 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_6(c_void_p(buf37.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf37, (12, 512, 64), (32768, 64, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_168, reinterpret_tensor(buf38, (12, 512, 64), (64, 768, 1), 0), out=buf41)
    del permute_168
    buf42 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf38, (12, 512, 64), (64, 768, 1), 0), permute_169, out=buf42)
    del permute_169
    buf43 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf44 = reinterpret_tensor(buf42, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf42  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_7(c_void_p(buf44.data_ptr()), c_void_p(convert_element_type_46.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(buf43.data_ptr()))
    del alias_43
    del convert_element_type_46
    buf45 = reinterpret_tensor(buf38, (12, 64, 512), (32768, 512, 1), 0); del buf38  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_170, reinterpret_tensor(buf44, (12, 512, 512), (262144, 512, 1), 0), out=buf45)
    del permute_170
    buf46 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf44, (12, 512, 512), (262144, 512, 1), 0), permute_171, out=buf46)
    del permute_171
    buf47 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf49 = empty((1, 512, 12, 192), device='cpu', dtype=torch.float32)
    cpp_fused_clone_div_sqrt_sum_8(c_void_p(buf41.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    buf50 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (2304, 512), (1, 2304), 0), view_198, out=buf50)
    del view_198
    buf51 = reinterpret_tensor(buf46, (512, 768), (768, 1), 0); del buf46  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (512, 2304), (2304, 1), 0), permute_178, out=buf51)
    del permute_178
    buf52 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf53 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf54 = buf35; del buf35  # reuse
    buf55 = buf34; del buf34  # reuse
    buf56 = buf33; del buf33  # reuse
    buf57 = buf36; del buf36  # reuse
    buf58 = reinterpret_tensor(buf45, (1, 512, 768), (393216, 768, 1), 0); del buf45  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_9(c_void_p(buf57.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(sub_89.data_ptr()), c_void_p(sqrt_33.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(convert_element_type_44.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()))
    del convert_element_type_44
    del primals_67
    del sqrt_33
    del sub_89
    buf59 = reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (512, 768), (768, 1), 0), permute_180, out=buf59)
    del permute_180
    buf60 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (768, 512), (1, 768), 0), view_196, out=buf60)
    del view_196
    buf61 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf59, (1, 512, 3072), (1572864, 3072, 1), 0); del buf59  # reuse
    cpp_fused_gelu_gelu_backward_sum_10(c_void_p(buf62.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(buf61.data_ptr()))
    del addmm_31
    buf63 = reinterpret_tensor(buf58, (512, 768), (768, 1), 0); del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (512, 3072), (3072, 1), 0), permute_184, out=buf63)
    del permute_184
    buf64 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (3072, 512), (1, 3072), 0), view_194, out=buf64)
    del view_194
    buf65 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf66 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf67 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf68 = buf56; del buf56  # reuse
    buf69 = buf55; del buf55  # reuse
    buf70 = buf54; del buf54  # reuse
    buf71 = buf57; del buf57  # reuse
    buf72 = reinterpret_tensor(buf51, (1, 512, 768), (393216, 768, 1), 0); del buf51  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_11(c_void_p(buf71.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(sub_86.data_ptr()), c_void_p(sqrt_32.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(convert_element_type_43.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    del convert_element_type_43
    del primals_65
    del sqrt_32
    del sub_86
    buf73 = buf63; del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (512, 768), (768, 1), 0), permute_188, out=buf73)
    del permute_188
    buf74 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (768, 512), (1, 768), 0), view_192, out=buf74)
    del view_192
    buf75 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_12(c_void_p(buf72.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = reinterpret_tensor(buf72, (12, 512, 64), (32768, 64, 1), 0); del buf72  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_193, reinterpret_tensor(buf73, (12, 512, 64), (64, 768, 1), 0), out=buf76)
    del permute_193
    buf77 = reinterpret_tensor(buf44, (12, 512, 512), (262144, 512, 1), 0); del buf44  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf73, (12, 512, 64), (64, 768, 1), 0), permute_194, out=buf77)
    del permute_194
    buf78 = buf43; del buf43  # reuse
    buf79 = reinterpret_tensor(buf77, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf77  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_13(c_void_p(buf79.data_ptr()), c_void_p(convert_element_type_42.data_ptr()), c_void_p(alias_48.data_ptr()), c_void_p(buf78.data_ptr()))
    del alias_48
    del convert_element_type_42
    buf80 = reinterpret_tensor(buf73, (12, 64, 512), (32768, 512, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_195, reinterpret_tensor(buf79, (12, 512, 512), (262144, 512, 1), 0), out=buf80)
    del permute_195
    buf81 = buf41; del buf41  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf79, (12, 512, 512), (262144, 512, 1), 0), permute_196, out=buf81)
    del permute_196
    buf82 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf83 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf84 = buf49; del buf49  # reuse
    cpp_fused_clone_div_sqrt_sum_14(c_void_p(buf76.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (2304, 512), (1, 2304), 0), view_180, out=buf85)
    del view_180
    buf86 = reinterpret_tensor(buf81, (512, 768), (768, 1), 0); del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (512, 2304), (2304, 1), 0), permute_203, out=buf86)
    del permute_203
    buf87 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf88 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf89 = buf70; del buf70  # reuse
    buf90 = buf69; del buf69  # reuse
    buf91 = buf68; del buf68  # reuse
    buf92 = buf71; del buf71  # reuse
    buf93 = reinterpret_tensor(buf80, (1, 512, 768), (393216, 768, 1), 0); del buf80  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_15(c_void_p(buf92.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(sub_81.data_ptr()), c_void_p(sqrt_30.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(convert_element_type_40.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    del convert_element_type_40
    del primals_61
    del sqrt_30
    del sub_81
    buf94 = reinterpret_tensor(buf62, (512, 3072), (3072, 1), 0); del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (512, 768), (768, 1), 0), permute_205, out=buf94)
    del permute_205
    buf95 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (768, 512), (1, 768), 0), view_178, out=buf95)
    del view_178
    buf96 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf97 = reinterpret_tensor(buf94, (1, 512, 3072), (1572864, 3072, 1), 0); del buf94  # reuse
    cpp_fused_gelu_gelu_backward_sum_16(c_void_p(buf97.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf96.data_ptr()))
    del addmm_28
    buf98 = reinterpret_tensor(buf93, (512, 768), (768, 1), 0); del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (512, 3072), (3072, 1), 0), permute_209, out=buf98)
    del permute_209
    buf99 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (3072, 512), (1, 3072), 0), view_176, out=buf99)
    del view_176
    buf100 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf101 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf102 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf103 = buf91; del buf91  # reuse
    buf104 = buf90; del buf90  # reuse
    buf105 = buf89; del buf89  # reuse
    buf106 = buf92; del buf92  # reuse
    buf107 = reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0); del buf86  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_17(c_void_p(buf106.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(sub_78.data_ptr()), c_void_p(sqrt_29.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(convert_element_type_39.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()))
    del convert_element_type_39
    del primals_59
    del sqrt_29
    del sub_78
    buf108 = buf98; del buf98  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (512, 768), (768, 1), 0), permute_213, out=buf108)
    del permute_213
    buf109 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (768, 512), (1, 768), 0), view_174, out=buf109)
    del view_174
    buf110 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_18(c_void_p(buf107.data_ptr()), c_void_p(buf110.data_ptr()))
    buf111 = reinterpret_tensor(buf107, (12, 512, 64), (32768, 64, 1), 0); del buf107  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_218, reinterpret_tensor(buf108, (12, 512, 64), (64, 768, 1), 0), out=buf111)
    del permute_218
    buf112 = reinterpret_tensor(buf79, (12, 512, 512), (262144, 512, 1), 0); del buf79  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (12, 512, 64), (64, 768, 1), 0), permute_219, out=buf112)
    del permute_219
    buf113 = buf78; del buf78  # reuse
    buf114 = reinterpret_tensor(buf112, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf112  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_19(c_void_p(buf114.data_ptr()), c_void_p(convert_element_type_38.data_ptr()), c_void_p(alias_53.data_ptr()), c_void_p(buf113.data_ptr()))
    del alias_53
    del convert_element_type_38
    buf115 = reinterpret_tensor(buf108, (12, 64, 512), (32768, 512, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_220, reinterpret_tensor(buf114, (12, 512, 512), (262144, 512, 1), 0), out=buf115)
    del permute_220
    buf116 = buf76; del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf114, (12, 512, 512), (262144, 512, 1), 0), permute_221, out=buf116)
    del permute_221
    buf117 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf118 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf119 = buf84; del buf84  # reuse
    cpp_fused_clone_div_sqrt_sum_20(c_void_p(buf111.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (2304, 512), (1, 2304), 0), view_162, out=buf120)
    del view_162
    buf121 = reinterpret_tensor(buf116, (512, 768), (768, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (512, 2304), (2304, 1), 0), permute_228, out=buf121)
    del permute_228
    buf122 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf123 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf124 = buf105; del buf105  # reuse
    buf125 = buf104; del buf104  # reuse
    buf126 = buf103; del buf103  # reuse
    buf127 = buf106; del buf106  # reuse
    buf128 = reinterpret_tensor(buf115, (1, 512, 768), (393216, 768, 1), 0); del buf115  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_21(c_void_p(buf127.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(sub_73.data_ptr()), c_void_p(sqrt_27.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(convert_element_type_36.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    del convert_element_type_36
    del primals_55
    del sqrt_27
    del sub_73
    buf129 = reinterpret_tensor(buf97, (512, 3072), (3072, 1), 0); del buf97  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (512, 768), (768, 1), 0), permute_230, out=buf129)
    del permute_230
    buf130 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (768, 512), (1, 768), 0), view_160, out=buf130)
    del view_160
    buf131 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf132 = reinterpret_tensor(buf129, (1, 512, 3072), (1572864, 3072, 1), 0); del buf129  # reuse
    cpp_fused_gelu_gelu_backward_sum_22(c_void_p(buf132.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(buf131.data_ptr()))
    del addmm_25
    buf133 = reinterpret_tensor(buf128, (512, 768), (768, 1), 0); del buf128  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (512, 3072), (3072, 1), 0), permute_234, out=buf133)
    del permute_234
    buf134 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (3072, 512), (1, 3072), 0), view_158, out=buf134)
    del view_158
    buf135 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf136 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf137 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf138 = buf126; del buf126  # reuse
    buf139 = buf125; del buf125  # reuse
    buf140 = buf124; del buf124  # reuse
    buf141 = buf127; del buf127  # reuse
    buf142 = reinterpret_tensor(buf121, (1, 512, 768), (393216, 768, 1), 0); del buf121  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_23(c_void_p(buf141.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(sub_70.data_ptr()), c_void_p(sqrt_26.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(convert_element_type_35.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()))
    del convert_element_type_35
    del primals_53
    del sqrt_26
    del sub_70
    buf143 = buf133; del buf133  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (512, 768), (768, 1), 0), permute_238, out=buf143)
    del permute_238
    buf144 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (768, 512), (1, 768), 0), view_156, out=buf144)
    del view_156
    buf145 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_24(c_void_p(buf142.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = reinterpret_tensor(buf142, (12, 512, 64), (32768, 64, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_243, reinterpret_tensor(buf143, (12, 512, 64), (64, 768, 1), 0), out=buf146)
    del permute_243
    buf147 = reinterpret_tensor(buf114, (12, 512, 512), (262144, 512, 1), 0); del buf114  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf143, (12, 512, 64), (64, 768, 1), 0), permute_244, out=buf147)
    del permute_244
    buf148 = buf113; del buf113  # reuse
    buf149 = reinterpret_tensor(buf147, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf147  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_25(c_void_p(buf149.data_ptr()), c_void_p(convert_element_type_34.data_ptr()), c_void_p(alias_58.data_ptr()), c_void_p(buf148.data_ptr()))
    del alias_58
    del convert_element_type_34
    buf150 = reinterpret_tensor(buf143, (12, 64, 512), (32768, 512, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_245, reinterpret_tensor(buf149, (12, 512, 512), (262144, 512, 1), 0), out=buf150)
    del permute_245
    buf151 = buf111; del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (12, 512, 512), (262144, 512, 1), 0), permute_246, out=buf151)
    del permute_246
    buf152 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf153 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf154 = buf119; del buf119  # reuse
    cpp_fused_clone_div_sqrt_sum_26(c_void_p(buf146.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (2304, 512), (1, 2304), 0), view_144, out=buf155)
    del view_144
    buf156 = reinterpret_tensor(buf151, (512, 768), (768, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (512, 2304), (2304, 1), 0), permute_253, out=buf156)
    del permute_253
    buf157 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf158 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf159 = buf140; del buf140  # reuse
    buf160 = buf139; del buf139  # reuse
    buf161 = buf138; del buf138  # reuse
    buf162 = buf141; del buf141  # reuse
    buf163 = reinterpret_tensor(buf150, (1, 512, 768), (393216, 768, 1), 0); del buf150  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_27(c_void_p(buf162.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(sub_65.data_ptr()), c_void_p(sqrt_24.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(convert_element_type_32.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    del convert_element_type_32
    del primals_49
    del sqrt_24
    del sub_65
    buf164 = reinterpret_tensor(buf132, (512, 3072), (3072, 1), 0); del buf132  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (512, 768), (768, 1), 0), permute_255, out=buf164)
    del permute_255
    buf165 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (768, 512), (1, 768), 0), view_142, out=buf165)
    del view_142
    buf166 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf167 = reinterpret_tensor(buf164, (1, 512, 3072), (1572864, 3072, 1), 0); del buf164  # reuse
    cpp_fused_gelu_gelu_backward_sum_28(c_void_p(buf167.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf166.data_ptr()))
    del addmm_22
    buf168 = reinterpret_tensor(buf163, (512, 768), (768, 1), 0); del buf163  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (512, 3072), (3072, 1), 0), permute_259, out=buf168)
    del permute_259
    buf169 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (3072, 512), (1, 3072), 0), view_140, out=buf169)
    del view_140
    buf170 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf171 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf172 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf173 = buf161; del buf161  # reuse
    buf174 = buf160; del buf160  # reuse
    buf175 = buf159; del buf159  # reuse
    buf176 = buf162; del buf162  # reuse
    buf177 = reinterpret_tensor(buf156, (1, 512, 768), (393216, 768, 1), 0); del buf156  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_29(c_void_p(buf176.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(sub_62.data_ptr()), c_void_p(sqrt_23.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(convert_element_type_31.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()))
    del convert_element_type_31
    del primals_47
    del sqrt_23
    del sub_62
    buf178 = buf168; del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (512, 768), (768, 1), 0), permute_263, out=buf178)
    del permute_263
    buf179 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (768, 512), (1, 768), 0), view_138, out=buf179)
    del view_138
    buf180 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_30(c_void_p(buf177.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf177, (12, 512, 64), (32768, 64, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_268, reinterpret_tensor(buf178, (12, 512, 64), (64, 768, 1), 0), out=buf181)
    del permute_268
    buf182 = reinterpret_tensor(buf149, (12, 512, 512), (262144, 512, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf178, (12, 512, 64), (64, 768, 1), 0), permute_269, out=buf182)
    del permute_269
    buf183 = buf148; del buf148  # reuse
    buf184 = reinterpret_tensor(buf182, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf182  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_31(c_void_p(buf184.data_ptr()), c_void_p(convert_element_type_30.data_ptr()), c_void_p(alias_63.data_ptr()), c_void_p(buf183.data_ptr()))
    del alias_63
    del convert_element_type_30
    buf185 = reinterpret_tensor(buf178, (12, 64, 512), (32768, 512, 1), 0); del buf178  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_270, reinterpret_tensor(buf184, (12, 512, 512), (262144, 512, 1), 0), out=buf185)
    del permute_270
    buf186 = buf146; del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (12, 512, 512), (262144, 512, 1), 0), permute_271, out=buf186)
    del permute_271
    buf187 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf188 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf189 = buf154; del buf154  # reuse
    cpp_fused_clone_div_sqrt_sum_32(c_void_p(buf181.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (2304, 512), (1, 2304), 0), view_126, out=buf190)
    del view_126
    buf191 = reinterpret_tensor(buf186, (512, 768), (768, 1), 0); del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (512, 2304), (2304, 1), 0), permute_278, out=buf191)
    del permute_278
    buf192 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf193 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf194 = buf175; del buf175  # reuse
    buf195 = buf174; del buf174  # reuse
    buf196 = buf173; del buf173  # reuse
    buf197 = buf176; del buf176  # reuse
    buf198 = reinterpret_tensor(buf185, (1, 512, 768), (393216, 768, 1), 0); del buf185  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_33(c_void_p(buf197.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(sub_57.data_ptr()), c_void_p(sqrt_21.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(convert_element_type_28.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()))
    del convert_element_type_28
    del primals_43
    del sqrt_21
    del sub_57
    buf199 = reinterpret_tensor(buf167, (512, 3072), (3072, 1), 0); del buf167  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (512, 768), (768, 1), 0), permute_280, out=buf199)
    del permute_280
    buf200 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (768, 512), (1, 768), 0), view_124, out=buf200)
    del view_124
    buf201 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf202 = reinterpret_tensor(buf199, (1, 512, 3072), (1572864, 3072, 1), 0); del buf199  # reuse
    cpp_fused_gelu_gelu_backward_sum_34(c_void_p(buf202.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(buf201.data_ptr()))
    del addmm_19
    buf203 = reinterpret_tensor(buf198, (512, 768), (768, 1), 0); del buf198  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (512, 3072), (3072, 1), 0), permute_284, out=buf203)
    del permute_284
    buf204 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (3072, 512), (1, 3072), 0), view_122, out=buf204)
    del view_122
    buf205 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf206 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf207 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf208 = buf196; del buf196  # reuse
    buf209 = buf195; del buf195  # reuse
    buf210 = buf194; del buf194  # reuse
    buf211 = buf197; del buf197  # reuse
    buf212 = reinterpret_tensor(buf191, (1, 512, 768), (393216, 768, 1), 0); del buf191  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_35(c_void_p(buf211.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(sub_54.data_ptr()), c_void_p(sqrt_20.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(convert_element_type_27.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()))
    del convert_element_type_27
    del primals_41
    del sqrt_20
    del sub_54
    buf213 = buf203; del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (512, 768), (768, 1), 0), permute_288, out=buf213)
    del permute_288
    buf214 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (768, 512), (1, 768), 0), view_120, out=buf214)
    del view_120
    buf215 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_36(c_void_p(buf212.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = reinterpret_tensor(buf212, (12, 512, 64), (32768, 64, 1), 0); del buf212  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_293, reinterpret_tensor(buf213, (12, 512, 64), (64, 768, 1), 0), out=buf216)
    del permute_293
    buf217 = reinterpret_tensor(buf184, (12, 512, 512), (262144, 512, 1), 0); del buf184  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf213, (12, 512, 64), (64, 768, 1), 0), permute_294, out=buf217)
    del permute_294
    buf218 = buf183; del buf183  # reuse
    buf219 = reinterpret_tensor(buf217, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf217  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_37(c_void_p(buf219.data_ptr()), c_void_p(convert_element_type_26.data_ptr()), c_void_p(alias_68.data_ptr()), c_void_p(buf218.data_ptr()))
    del alias_68
    del convert_element_type_26
    buf220 = reinterpret_tensor(buf213, (12, 64, 512), (32768, 512, 1), 0); del buf213  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_295, reinterpret_tensor(buf219, (12, 512, 512), (262144, 512, 1), 0), out=buf220)
    del permute_295
    buf221 = buf181; del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf219, (12, 512, 512), (262144, 512, 1), 0), permute_296, out=buf221)
    del permute_296
    buf222 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf224 = buf189; del buf189  # reuse
    cpp_fused_clone_div_sqrt_sum_38(c_void_p(buf216.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    buf225 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (2304, 512), (1, 2304), 0), view_108, out=buf225)
    del view_108
    buf226 = reinterpret_tensor(buf221, (512, 768), (768, 1), 0); del buf221  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (512, 2304), (2304, 1), 0), permute_303, out=buf226)
    del permute_303
    buf227 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf228 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf229 = buf210; del buf210  # reuse
    buf230 = buf209; del buf209  # reuse
    buf231 = buf208; del buf208  # reuse
    buf232 = buf211; del buf211  # reuse
    buf233 = reinterpret_tensor(buf220, (1, 512, 768), (393216, 768, 1), 0); del buf220  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_39(c_void_p(buf232.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(sub_49.data_ptr()), c_void_p(sqrt_18.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(convert_element_type_24.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()))
    del convert_element_type_24
    del primals_37
    del sqrt_18
    del sub_49
    buf234 = reinterpret_tensor(buf202, (512, 3072), (3072, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (512, 768), (768, 1), 0), permute_305, out=buf234)
    del permute_305
    buf235 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (768, 512), (1, 768), 0), view_106, out=buf235)
    del view_106
    buf236 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf237 = reinterpret_tensor(buf234, (1, 512, 3072), (1572864, 3072, 1), 0); del buf234  # reuse
    cpp_fused_gelu_gelu_backward_sum_40(c_void_p(buf237.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf236.data_ptr()))
    del addmm_16
    buf238 = reinterpret_tensor(buf233, (512, 768), (768, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (512, 3072), (3072, 1), 0), permute_309, out=buf238)
    del permute_309
    buf239 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (3072, 512), (1, 3072), 0), view_104, out=buf239)
    del view_104
    buf240 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf241 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf242 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf243 = buf231; del buf231  # reuse
    buf244 = buf230; del buf230  # reuse
    buf245 = buf229; del buf229  # reuse
    buf246 = buf232; del buf232  # reuse
    buf247 = reinterpret_tensor(buf226, (1, 512, 768), (393216, 768, 1), 0); del buf226  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_41(c_void_p(buf246.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(sub_46.data_ptr()), c_void_p(sqrt_17.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(convert_element_type_23.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del convert_element_type_23
    del primals_35
    del sqrt_17
    del sub_46
    buf248 = buf238; del buf238  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (512, 768), (768, 1), 0), permute_313, out=buf248)
    del permute_313
    buf249 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (768, 512), (1, 768), 0), view_102, out=buf249)
    del view_102
    buf250 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_42(c_void_p(buf247.data_ptr()), c_void_p(buf250.data_ptr()))
    buf251 = reinterpret_tensor(buf247, (12, 512, 64), (32768, 64, 1), 0); del buf247  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_318, reinterpret_tensor(buf248, (12, 512, 64), (64, 768, 1), 0), out=buf251)
    del permute_318
    buf252 = reinterpret_tensor(buf219, (12, 512, 512), (262144, 512, 1), 0); del buf219  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf248, (12, 512, 64), (64, 768, 1), 0), permute_319, out=buf252)
    del permute_319
    buf253 = buf218; del buf218  # reuse
    buf254 = reinterpret_tensor(buf252, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf252  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_43(c_void_p(buf254.data_ptr()), c_void_p(convert_element_type_22.data_ptr()), c_void_p(alias_73.data_ptr()), c_void_p(buf253.data_ptr()))
    del alias_73
    del convert_element_type_22
    buf255 = reinterpret_tensor(buf248, (12, 64, 512), (32768, 512, 1), 0); del buf248  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_320, reinterpret_tensor(buf254, (12, 512, 512), (262144, 512, 1), 0), out=buf255)
    del permute_320
    buf256 = buf216; del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf254, (12, 512, 512), (262144, 512, 1), 0), permute_321, out=buf256)
    del permute_321
    buf257 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf258 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf259 = buf224; del buf224  # reuse
    cpp_fused_clone_div_sqrt_sum_44(c_void_p(buf251.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    buf260 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (2304, 512), (1, 2304), 0), view_90, out=buf260)
    del view_90
    buf261 = reinterpret_tensor(buf256, (512, 768), (768, 1), 0); del buf256  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (512, 2304), (2304, 1), 0), permute_328, out=buf261)
    del permute_328
    buf262 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf263 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf264 = buf245; del buf245  # reuse
    buf265 = buf244; del buf244  # reuse
    buf266 = buf243; del buf243  # reuse
    buf267 = buf246; del buf246  # reuse
    buf268 = reinterpret_tensor(buf255, (1, 512, 768), (393216, 768, 1), 0); del buf255  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_45(c_void_p(buf267.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(sub_41.data_ptr()), c_void_p(sqrt_15.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(convert_element_type_20.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()))
    del convert_element_type_20
    del primals_31
    del sqrt_15
    del sub_41
    buf269 = reinterpret_tensor(buf237, (512, 3072), (3072, 1), 0); del buf237  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf268, (512, 768), (768, 1), 0), permute_330, out=buf269)
    del permute_330
    buf270 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf268, (768, 512), (1, 768), 0), view_88, out=buf270)
    del view_88
    buf271 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf272 = reinterpret_tensor(buf269, (1, 512, 3072), (1572864, 3072, 1), 0); del buf269  # reuse
    cpp_fused_gelu_gelu_backward_sum_46(c_void_p(buf272.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(buf271.data_ptr()))
    del addmm_13
    buf273 = reinterpret_tensor(buf268, (512, 768), (768, 1), 0); del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (512, 3072), (3072, 1), 0), permute_334, out=buf273)
    del permute_334
    buf274 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (3072, 512), (1, 3072), 0), view_86, out=buf274)
    del view_86
    buf275 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf276 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf277 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf278 = buf266; del buf266  # reuse
    buf279 = buf265; del buf265  # reuse
    buf280 = buf264; del buf264  # reuse
    buf281 = buf267; del buf267  # reuse
    buf282 = reinterpret_tensor(buf261, (1, 512, 768), (393216, 768, 1), 0); del buf261  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_47(c_void_p(buf281.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(sub_38.data_ptr()), c_void_p(sqrt_14.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(convert_element_type_19.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()))
    del convert_element_type_19
    del primals_29
    del sqrt_14
    del sub_38
    buf283 = buf273; del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (512, 768), (768, 1), 0), permute_338, out=buf283)
    del permute_338
    buf284 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (768, 512), (1, 768), 0), view_84, out=buf284)
    del view_84
    buf285 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_48(c_void_p(buf282.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = reinterpret_tensor(buf282, (12, 512, 64), (32768, 64, 1), 0); del buf282  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_343, reinterpret_tensor(buf283, (12, 512, 64), (64, 768, 1), 0), out=buf286)
    del permute_343
    buf287 = reinterpret_tensor(buf254, (12, 512, 512), (262144, 512, 1), 0); del buf254  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf283, (12, 512, 64), (64, 768, 1), 0), permute_344, out=buf287)
    del permute_344
    buf288 = buf253; del buf253  # reuse
    buf289 = reinterpret_tensor(buf287, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf287  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_49(c_void_p(buf289.data_ptr()), c_void_p(convert_element_type_18.data_ptr()), c_void_p(alias_78.data_ptr()), c_void_p(buf288.data_ptr()))
    del alias_78
    del convert_element_type_18
    buf290 = reinterpret_tensor(buf283, (12, 64, 512), (32768, 512, 1), 0); del buf283  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_345, reinterpret_tensor(buf289, (12, 512, 512), (262144, 512, 1), 0), out=buf290)
    del permute_345
    buf291 = buf251; del buf251  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf289, (12, 512, 512), (262144, 512, 1), 0), permute_346, out=buf291)
    del permute_346
    buf292 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf293 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf294 = buf259; del buf259  # reuse
    cpp_fused_clone_div_sqrt_sum_50(c_void_p(buf286.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    buf295 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (2304, 512), (1, 2304), 0), view_72, out=buf295)
    del view_72
    buf296 = reinterpret_tensor(buf291, (512, 768), (768, 1), 0); del buf291  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (512, 2304), (2304, 1), 0), permute_353, out=buf296)
    del permute_353
    buf297 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf298 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf299 = buf280; del buf280  # reuse
    buf300 = buf279; del buf279  # reuse
    buf301 = buf278; del buf278  # reuse
    buf302 = buf281; del buf281  # reuse
    buf303 = reinterpret_tensor(buf290, (1, 512, 768), (393216, 768, 1), 0); del buf290  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_51(c_void_p(buf302.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(sub_33.data_ptr()), c_void_p(sqrt_12.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(convert_element_type_16.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf303.data_ptr()))
    del convert_element_type_16
    del primals_25
    del sqrt_12
    del sub_33
    buf304 = reinterpret_tensor(buf272, (512, 3072), (3072, 1), 0); del buf272  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (512, 768), (768, 1), 0), permute_355, out=buf304)
    del permute_355
    buf305 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (768, 512), (1, 768), 0), view_70, out=buf305)
    del view_70
    buf306 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf307 = reinterpret_tensor(buf304, (1, 512, 3072), (1572864, 3072, 1), 0); del buf304  # reuse
    cpp_fused_gelu_gelu_backward_sum_52(c_void_p(buf307.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf306.data_ptr()))
    del addmm_10
    buf308 = reinterpret_tensor(buf303, (512, 768), (768, 1), 0); del buf303  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (512, 3072), (3072, 1), 0), permute_359, out=buf308)
    del permute_359
    buf309 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (3072, 512), (1, 3072), 0), view_68, out=buf309)
    del view_68
    buf310 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf311 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf312 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf313 = buf301; del buf301  # reuse
    buf314 = buf300; del buf300  # reuse
    buf315 = buf299; del buf299  # reuse
    buf316 = buf302; del buf302  # reuse
    buf317 = reinterpret_tensor(buf296, (1, 512, 768), (393216, 768, 1), 0); del buf296  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_53(c_void_p(buf316.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(sub_30.data_ptr()), c_void_p(sqrt_11.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(convert_element_type_15.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()))
    del convert_element_type_15
    del primals_23
    del sqrt_11
    del sub_30
    buf318 = buf308; del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (512, 768), (768, 1), 0), permute_363, out=buf318)
    del permute_363
    buf319 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (768, 512), (1, 768), 0), view_66, out=buf319)
    del view_66
    buf320 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf317.data_ptr()), c_void_p(buf320.data_ptr()))
    buf321 = reinterpret_tensor(buf317, (12, 512, 64), (32768, 64, 1), 0); del buf317  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_368, reinterpret_tensor(buf318, (12, 512, 64), (64, 768, 1), 0), out=buf321)
    del permute_368
    buf322 = reinterpret_tensor(buf289, (12, 512, 512), (262144, 512, 1), 0); del buf289  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf318, (12, 512, 64), (64, 768, 1), 0), permute_369, out=buf322)
    del permute_369
    buf323 = buf288; del buf288  # reuse
    buf324 = reinterpret_tensor(buf322, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf322  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_55(c_void_p(buf324.data_ptr()), c_void_p(convert_element_type_14.data_ptr()), c_void_p(alias_83.data_ptr()), c_void_p(buf323.data_ptr()))
    del alias_83
    del convert_element_type_14
    buf325 = reinterpret_tensor(buf318, (12, 64, 512), (32768, 512, 1), 0); del buf318  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_370, reinterpret_tensor(buf324, (12, 512, 512), (262144, 512, 1), 0), out=buf325)
    del permute_370
    buf326 = buf286; del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf324, (12, 512, 512), (262144, 512, 1), 0), permute_371, out=buf326)
    del permute_371
    buf327 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf328 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf329 = buf294; del buf294  # reuse
    cpp_fused_clone_div_sqrt_sum_56(c_void_p(buf321.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    buf330 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf329, (2304, 512), (1, 2304), 0), view_54, out=buf330)
    del view_54
    buf331 = reinterpret_tensor(buf326, (512, 768), (768, 1), 0); del buf326  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf329, (512, 2304), (2304, 1), 0), permute_378, out=buf331)
    del permute_378
    buf332 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf333 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf334 = buf315; del buf315  # reuse
    buf335 = buf314; del buf314  # reuse
    buf336 = buf313; del buf313  # reuse
    buf337 = buf316; del buf316  # reuse
    buf338 = reinterpret_tensor(buf325, (1, 512, 768), (393216, 768, 1), 0); del buf325  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_57(c_void_p(buf337.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(sub_25.data_ptr()), c_void_p(sqrt_9.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(convert_element_type_12.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()))
    del convert_element_type_12
    del primals_19
    del sqrt_9
    del sub_25
    buf339 = reinterpret_tensor(buf307, (512, 3072), (3072, 1), 0); del buf307  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf338, (512, 768), (768, 1), 0), permute_380, out=buf339)
    del permute_380
    buf340 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf338, (768, 512), (1, 768), 0), view_52, out=buf340)
    del view_52
    buf341 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf342 = reinterpret_tensor(buf339, (1, 512, 3072), (1572864, 3072, 1), 0); del buf339  # reuse
    cpp_fused_gelu_gelu_backward_sum_58(c_void_p(buf342.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(buf341.data_ptr()))
    del addmm_7
    buf343 = reinterpret_tensor(buf338, (512, 768), (768, 1), 0); del buf338  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf342, (512, 3072), (3072, 1), 0), permute_384, out=buf343)
    del permute_384
    buf344 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf342, (3072, 512), (1, 3072), 0), view_50, out=buf344)
    del view_50
    buf345 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf346 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf347 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf348 = buf336; del buf336  # reuse
    buf349 = buf335; del buf335  # reuse
    buf350 = buf334; del buf334  # reuse
    buf351 = buf337; del buf337  # reuse
    buf352 = reinterpret_tensor(buf331, (1, 512, 768), (393216, 768, 1), 0); del buf331  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_59(c_void_p(buf351.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(sub_22.data_ptr()), c_void_p(sqrt_8.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(convert_element_type_11.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()))
    del convert_element_type_11
    del primals_17
    del sqrt_8
    del sub_22
    buf353 = buf343; del buf343  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (512, 768), (768, 1), 0), permute_388, out=buf353)
    del permute_388
    buf354 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (768, 512), (1, 768), 0), view_48, out=buf354)
    del view_48
    buf355 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_60(c_void_p(buf352.data_ptr()), c_void_p(buf355.data_ptr()))
    buf356 = reinterpret_tensor(buf352, (12, 512, 64), (32768, 64, 1), 0); del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_393, reinterpret_tensor(buf353, (12, 512, 64), (64, 768, 1), 0), out=buf356)
    del permute_393
    buf357 = reinterpret_tensor(buf324, (12, 512, 512), (262144, 512, 1), 0); del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf353, (12, 512, 64), (64, 768, 1), 0), permute_394, out=buf357)
    del permute_394
    buf358 = buf323; del buf323  # reuse
    buf359 = reinterpret_tensor(buf357, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf357  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_61(c_void_p(buf359.data_ptr()), c_void_p(convert_element_type_10.data_ptr()), c_void_p(alias_88.data_ptr()), c_void_p(buf358.data_ptr()))
    del alias_88
    del convert_element_type_10
    buf360 = reinterpret_tensor(buf353, (12, 64, 512), (32768, 512, 1), 0); del buf353  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_395, reinterpret_tensor(buf359, (12, 512, 512), (262144, 512, 1), 0), out=buf360)
    del permute_395
    buf361 = buf321; del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf359, (12, 512, 512), (262144, 512, 1), 0), permute_396, out=buf361)
    del permute_396
    buf362 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf363 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf364 = buf329; del buf329  # reuse
    cpp_fused_clone_div_sqrt_sum_62(c_void_p(buf356.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()))
    buf365 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (2304, 512), (1, 2304), 0), view_36, out=buf365)
    del view_36
    buf366 = reinterpret_tensor(buf361, (512, 768), (768, 1), 0); del buf361  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (512, 2304), (2304, 1), 0), permute_403, out=buf366)
    del permute_403
    buf367 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf368 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf369 = buf350; del buf350  # reuse
    buf370 = buf349; del buf349  # reuse
    buf371 = buf348; del buf348  # reuse
    buf372 = buf351; del buf351  # reuse
    buf373 = reinterpret_tensor(buf360, (1, 512, 768), (393216, 768, 1), 0); del buf360  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_63(c_void_p(buf372.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(sub_17.data_ptr()), c_void_p(sqrt_6.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(convert_element_type_8.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf373.data_ptr()))
    del convert_element_type_8
    del primals_13
    del sqrt_6
    del sub_17
    buf374 = reinterpret_tensor(buf342, (512, 3072), (3072, 1), 0); del buf342  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf373, (512, 768), (768, 1), 0), permute_405, out=buf374)
    del permute_405
    buf375 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf373, (768, 512), (1, 768), 0), view_34, out=buf375)
    del view_34
    buf376 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf377 = reinterpret_tensor(buf374, (1, 512, 3072), (1572864, 3072, 1), 0); del buf374  # reuse
    cpp_fused_gelu_gelu_backward_sum_64(c_void_p(buf377.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf376.data_ptr()))
    del addmm_4
    buf378 = reinterpret_tensor(buf373, (512, 768), (768, 1), 0); del buf373  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (512, 3072), (3072, 1), 0), permute_409, out=buf378)
    del permute_409
    buf379 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (3072, 512), (1, 3072), 0), view_32, out=buf379)
    del view_32
    buf380 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf381 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf382 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf383 = buf371; del buf371  # reuse
    buf384 = buf370; del buf370  # reuse
    buf385 = buf369; del buf369  # reuse
    buf386 = buf372; del buf372  # reuse
    buf387 = reinterpret_tensor(buf366, (1, 512, 768), (393216, 768, 1), 0); del buf366  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_65(c_void_p(buf386.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(sub_14.data_ptr()), c_void_p(sqrt_5.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(convert_element_type_7.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf387.data_ptr()))
    del convert_element_type_7
    del primals_11
    del sqrt_5
    del sub_14
    buf388 = buf378; del buf378  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf387, (512, 768), (768, 1), 0), permute_413, out=buf388)
    del permute_413
    buf389 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf387, (768, 512), (1, 768), 0), view_30, out=buf389)
    del view_30
    buf390 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_66(c_void_p(buf387.data_ptr()), c_void_p(buf390.data_ptr()))
    buf391 = reinterpret_tensor(buf387, (12, 512, 64), (32768, 64, 1), 0); del buf387  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_418, reinterpret_tensor(buf388, (12, 512, 64), (64, 768, 1), 0), out=buf391)
    del permute_418
    buf392 = reinterpret_tensor(buf359, (12, 512, 512), (262144, 512, 1), 0); del buf359  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf388, (12, 512, 64), (64, 768, 1), 0), permute_419, out=buf392)
    del permute_419
    buf393 = buf358; del buf358  # reuse
    buf394 = reinterpret_tensor(buf392, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf392  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_67(c_void_p(buf394.data_ptr()), c_void_p(convert_element_type_6.data_ptr()), c_void_p(alias_93.data_ptr()), c_void_p(buf393.data_ptr()))
    del alias_93
    del convert_element_type_6
    buf395 = reinterpret_tensor(buf388, (12, 64, 512), (32768, 512, 1), 0); del buf388  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_420, reinterpret_tensor(buf394, (12, 512, 512), (262144, 512, 1), 0), out=buf395)
    del permute_420
    buf396 = buf356; del buf356  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf394, (12, 512, 512), (262144, 512, 1), 0), permute_421, out=buf396)
    del permute_421
    buf397 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf398 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf399 = buf364; del buf364  # reuse
    cpp_fused_clone_div_sqrt_sum_68(c_void_p(buf391.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    buf400 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (2304, 512), (1, 2304), 0), view_18, out=buf400)
    del view_18
    buf401 = reinterpret_tensor(buf396, (512, 768), (768, 1), 0); del buf396  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (512, 2304), (2304, 1), 0), permute_428, out=buf401)
    del permute_428
    buf402 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf403 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf404 = buf385; del buf385  # reuse
    buf405 = buf384; del buf384  # reuse
    buf406 = buf383; del buf383  # reuse
    buf407 = buf386; del buf386  # reuse
    buf408 = reinterpret_tensor(buf395, (1, 512, 768), (393216, 768, 1), 0); del buf395  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_69(c_void_p(buf407.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(sub_9.data_ptr()), c_void_p(sqrt_3.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(convert_element_type_4.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf408.data_ptr()))
    del convert_element_type_4
    del primals_7
    del sqrt_3
    del sub_9
    buf409 = reinterpret_tensor(buf377, (512, 3072), (3072, 1), 0); del buf377  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (512, 768), (768, 1), 0), permute_430, out=buf409)
    del permute_430
    buf410 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (768, 512), (1, 768), 0), view_16, out=buf410)
    del view_16
    buf411 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf412 = reinterpret_tensor(buf409, (1, 512, 3072), (1572864, 3072, 1), 0); del buf409  # reuse
    cpp_fused_gelu_gelu_backward_sum_70(c_void_p(buf412.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(buf411.data_ptr()))
    del addmm_1
    buf413 = reinterpret_tensor(buf408, (512, 768), (768, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf412, (512, 3072), (3072, 1), 0), permute_434, out=buf413)
    del permute_434
    buf414 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf412, (3072, 512), (1, 3072), 0), view_14, out=buf414)
    del view_14
    buf415 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf416 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf417 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf418 = buf406; del buf406  # reuse
    buf419 = buf405; del buf405  # reuse
    buf420 = buf404; del buf404  # reuse
    buf421 = buf407; del buf407  # reuse
    buf422 = reinterpret_tensor(buf401, (1, 512, 768), (393216, 768, 1), 0); del buf401  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_71(c_void_p(buf421.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(sub_6.data_ptr()), c_void_p(sqrt_2.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(convert_element_type_3.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()))
    del buf412
    del convert_element_type_3
    del primals_5
    del sqrt_2
    del sub_6
    buf423 = buf413; del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (512, 768), (768, 1), 0), permute_438, out=buf423)
    del permute_438
    buf424 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (768, 512), (1, 768), 0), view_12, out=buf424)
    del view_12
    buf425 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_72(c_void_p(buf422.data_ptr()), c_void_p(buf425.data_ptr()))
    buf426 = reinterpret_tensor(buf422, (12, 512, 64), (32768, 64, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_443, reinterpret_tensor(buf423, (12, 512, 64), (64, 768, 1), 0), out=buf426)
    del permute_443
    buf427 = reinterpret_tensor(buf394, (12, 512, 512), (262144, 512, 1), 0); del buf394  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf423, (12, 512, 64), (64, 768, 1), 0), permute_444, out=buf427)
    del permute_444
    buf428 = buf393; del buf393  # reuse
    buf429 = reinterpret_tensor(buf427, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf427  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_73(c_void_p(buf429.data_ptr()), c_void_p(convert_element_type_2.data_ptr()), c_void_p(alias_98.data_ptr()), c_void_p(buf428.data_ptr()))
    del alias_98
    del buf428
    del convert_element_type_2
    buf430 = reinterpret_tensor(buf423, (12, 64, 512), (32768, 512, 1), 0); del buf423  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_445, reinterpret_tensor(buf429, (12, 512, 512), (262144, 512, 1), 0), out=buf430)
    del permute_445
    buf431 = buf391; del buf391  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf429, (12, 512, 512), (262144, 512, 1), 0), permute_446, out=buf431)
    del buf429
    del permute_446
    buf432 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf433 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf434 = buf399; del buf399  # reuse
    cpp_fused_clone_div_sqrt_sum_74(c_void_p(buf426.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()))
    buf435 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf434, (2304, 512), (1, 2304), 0), view, out=buf435)
    del view
    buf436 = reinterpret_tensor(buf431, (512, 768), (768, 1), 0); del buf431  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf434, (512, 2304), (2304, 1), 0), permute_453, out=buf436)
    del buf434
    del permute_453
    buf437 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf438 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf439 = reinterpret_tensor(buf430, (1, 512, 768), (393216, 768, 1), 0); del buf430  # reuse
    buf440 = buf420; del buf420  # reuse
    buf441 = buf419; del buf419  # reuse
    buf442 = buf418; del buf418  # reuse
    buf443 = buf439; del buf439  # reuse
    buf445 = reinterpret_tensor(buf426, (1, 512, 768), (393216, 768, 1), 0); del buf426  # reuse
    buf449 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf444 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_75(c_void_p(buf443.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(sub.data_ptr()), c_void_p(sqrt.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(slice_1.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf444.data_ptr()))
    del buf421
    del buf436
    del buf440
    del buf441
    del buf442
    del buf443
    del convert_element_type
    del primals_1
    del sqrt
    del sub
    aten.index_put_(buf444, [slice_1], buf445, True)
    del buf445
    del slice_1
    buf448 = empty((50265, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_76(c_void_p(buf448.data_ptr()))
    aten.index_put_(buf448, [primals_168], buf449, True)
    del buf449
    del primals_168
    return (reinterpret_tensor(buf438, (768, ), (1, ), 0), reinterpret_tensor(buf437, (768, ), (1, ), 0), reinterpret_tensor(buf433, (768, ), (1, ), 0), reinterpret_tensor(buf432, (768, ), (1, ), 0), reinterpret_tensor(buf417, (768, ), (1, ), 0), reinterpret_tensor(buf416, (768, ), (1, ), 0), reinterpret_tensor(buf403, (768, ), (1, ), 0), reinterpret_tensor(buf402, (768, ), (1, ), 0), reinterpret_tensor(buf398, (768, ), (1, ), 0), reinterpret_tensor(buf397, (768, ), (1, ), 0), reinterpret_tensor(buf382, (768, ), (1, ), 0), reinterpret_tensor(buf381, (768, ), (1, ), 0), reinterpret_tensor(buf368, (768, ), (1, ), 0), reinterpret_tensor(buf367, (768, ), (1, ), 0), reinterpret_tensor(buf363, (768, ), (1, ), 0), reinterpret_tensor(buf362, (768, ), (1, ), 0), reinterpret_tensor(buf347, (768, ), (1, ), 0), reinterpret_tensor(buf346, (768, ), (1, ), 0), reinterpret_tensor(buf333, (768, ), (1, ), 0), reinterpret_tensor(buf332, (768, ), (1, ), 0), reinterpret_tensor(buf328, (768, ), (1, ), 0), reinterpret_tensor(buf327, (768, ), (1, ), 0), reinterpret_tensor(buf312, (768, ), (1, ), 0), reinterpret_tensor(buf311, (768, ), (1, ), 0), reinterpret_tensor(buf298, (768, ), (1, ), 0), reinterpret_tensor(buf297, (768, ), (1, ), 0), reinterpret_tensor(buf293, (768, ), (1, ), 0), reinterpret_tensor(buf292, (768, ), (1, ), 0), reinterpret_tensor(buf277, (768, ), (1, ), 0), reinterpret_tensor(buf276, (768, ), (1, ), 0), reinterpret_tensor(buf263, (768, ), (1, ), 0), reinterpret_tensor(buf262, (768, ), (1, ), 0), reinterpret_tensor(buf258, (768, ), (1, ), 0), reinterpret_tensor(buf257, (768, ), (1, ), 0), reinterpret_tensor(buf242, (768, ), (1, ), 0), reinterpret_tensor(buf241, (768, ), (1, ), 0), reinterpret_tensor(buf228, (768, ), (1, ), 0), reinterpret_tensor(buf227, (768, ), (1, ), 0), reinterpret_tensor(buf223, (768, ), (1, ), 0), reinterpret_tensor(buf222, (768, ), (1, ), 0), reinterpret_tensor(buf207, (768, ), (1, ), 0), reinterpret_tensor(buf206, (768, ), (1, ), 0), reinterpret_tensor(buf193, (768, ), (1, ), 0), reinterpret_tensor(buf192, (768, ), (1, ), 0), reinterpret_tensor(buf188, (768, ), (1, ), 0), reinterpret_tensor(buf187, (768, ), (1, ), 0), reinterpret_tensor(buf172, (768, ), (1, ), 0), reinterpret_tensor(buf171, (768, ), (1, ), 0), reinterpret_tensor(buf158, (768, ), (1, ), 0), reinterpret_tensor(buf157, (768, ), (1, ), 0), reinterpret_tensor(buf153, (768, ), (1, ), 0), reinterpret_tensor(buf152, (768, ), (1, ), 0), reinterpret_tensor(buf137, (768, ), (1, ), 0), reinterpret_tensor(buf136, (768, ), (1, ), 0), reinterpret_tensor(buf123, (768, ), (1, ), 0), reinterpret_tensor(buf122, (768, ), (1, ), 0), reinterpret_tensor(buf118, (768, ), (1, ), 0), reinterpret_tensor(buf117, (768, ), (1, ), 0), reinterpret_tensor(buf102, (768, ), (1, ), 0), reinterpret_tensor(buf101, (768, ), (1, ), 0), reinterpret_tensor(buf88, (768, ), (1, ), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), reinterpret_tensor(buf83, (768, ), (1, ), 0), reinterpret_tensor(buf82, (768, ), (1, ), 0), reinterpret_tensor(buf67, (768, ), (1, ), 0), reinterpret_tensor(buf66, (768, ), (1, ), 0), reinterpret_tensor(buf53, (768, ), (1, ), 0), reinterpret_tensor(buf52, (768, ), (1, ), 0), reinterpret_tensor(buf48, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, ), (1, ), 0), reinterpret_tensor(buf32, (768, ), (1, ), 0), reinterpret_tensor(buf31, (768, ), (1, ), 0), reinterpret_tensor(buf18, (768, ), (1, ), 0), reinterpret_tensor(buf17, (768, ), (1, ), 0), buf448, buf444, reinterpret_tensor(buf435, (2304, 768), (768, 1), 0), reinterpret_tensor(buf424, (768, 768), (768, 1), 0), reinterpret_tensor(buf425, (768, ), (1, ), 0), reinterpret_tensor(buf414, (3072, 768), (768, 1), 0), reinterpret_tensor(buf415, (3072, ), (1, ), 0), reinterpret_tensor(buf410, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf411, (768, ), (1, ), 0), reinterpret_tensor(buf400, (2304, 768), (768, 1), 0), reinterpret_tensor(buf389, (768, 768), (768, 1), 0), reinterpret_tensor(buf390, (768, ), (1, ), 0), reinterpret_tensor(buf379, (3072, 768), (768, 1), 0), reinterpret_tensor(buf380, (3072, ), (1, ), 0), reinterpret_tensor(buf375, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf376, (768, ), (1, ), 0), reinterpret_tensor(buf365, (2304, 768), (768, 1), 0), reinterpret_tensor(buf354, (768, 768), (768, 1), 0), reinterpret_tensor(buf355, (768, ), (1, ), 0), reinterpret_tensor(buf344, (3072, 768), (768, 1), 0), reinterpret_tensor(buf345, (3072, ), (1, ), 0), reinterpret_tensor(buf340, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf341, (768, ), (1, ), 0), reinterpret_tensor(buf330, (2304, 768), (768, 1), 0), reinterpret_tensor(buf319, (768, 768), (768, 1), 0), reinterpret_tensor(buf320, (768, ), (1, ), 0), reinterpret_tensor(buf309, (3072, 768), (768, 1), 0), reinterpret_tensor(buf310, (3072, ), (1, ), 0), reinterpret_tensor(buf305, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf306, (768, ), (1, ), 0), reinterpret_tensor(buf295, (2304, 768), (768, 1), 0), reinterpret_tensor(buf284, (768, 768), (768, 1), 0), reinterpret_tensor(buf285, (768, ), (1, ), 0), reinterpret_tensor(buf274, (3072, 768), (768, 1), 0), reinterpret_tensor(buf275, (3072, ), (1, ), 0), reinterpret_tensor(buf270, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf271, (768, ), (1, ), 0), reinterpret_tensor(buf260, (2304, 768), (768, 1), 0), reinterpret_tensor(buf249, (768, 768), (768, 1), 0), reinterpret_tensor(buf250, (768, ), (1, ), 0), reinterpret_tensor(buf239, (3072, 768), (768, 1), 0), reinterpret_tensor(buf240, (3072, ), (1, ), 0), reinterpret_tensor(buf235, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf236, (768, ), (1, ), 0), reinterpret_tensor(buf225, (2304, 768), (768, 1), 0), reinterpret_tensor(buf214, (768, 768), (768, 1), 0), reinterpret_tensor(buf215, (768, ), (1, ), 0), reinterpret_tensor(buf204, (3072, 768), (768, 1), 0), reinterpret_tensor(buf205, (3072, ), (1, ), 0), reinterpret_tensor(buf200, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf201, (768, ), (1, ), 0), reinterpret_tensor(buf190, (2304, 768), (768, 1), 0), reinterpret_tensor(buf179, (768, 768), (768, 1), 0), reinterpret_tensor(buf180, (768, ), (1, ), 0), reinterpret_tensor(buf169, (3072, 768), (768, 1), 0), reinterpret_tensor(buf170, (3072, ), (1, ), 0), reinterpret_tensor(buf165, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf166, (768, ), (1, ), 0), reinterpret_tensor(buf155, (2304, 768), (768, 1), 0), reinterpret_tensor(buf144, (768, 768), (768, 1), 0), reinterpret_tensor(buf145, (768, ), (1, ), 0), reinterpret_tensor(buf134, (3072, 768), (768, 1), 0), reinterpret_tensor(buf135, (3072, ), (1, ), 0), reinterpret_tensor(buf130, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf131, (768, ), (1, ), 0), reinterpret_tensor(buf120, (2304, 768), (768, 1), 0), reinterpret_tensor(buf109, (768, 768), (768, 1), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), reinterpret_tensor(buf99, (3072, 768), (768, 1), 0), reinterpret_tensor(buf100, (3072, ), (1, ), 0), reinterpret_tensor(buf95, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf96, (768, ), (1, ), 0), reinterpret_tensor(buf85, (2304, 768), (768, 1), 0), reinterpret_tensor(buf74, (768, 768), (768, 1), 0), reinterpret_tensor(buf75, (768, ), (1, ), 0), reinterpret_tensor(buf64, (3072, 768), (768, 1), 0), reinterpret_tensor(buf65, (3072, ), (1, ), 0), reinterpret_tensor(buf60, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf61, (768, ), (1, ), 0), reinterpret_tensor(buf50, (2304, 768), (768, 1), 0), reinterpret_tensor(buf39, (768, 768), (768, 1), 0), reinterpret_tensor(buf40, (768, ), (1, ), 0), reinterpret_tensor(buf29, (3072, 768), (768, 1), 0), reinterpret_tensor(buf30, (3072, ), (1, ), 0), reinterpret_tensor(buf25, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf26, (768, ), (1, ), 0), reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf16, (768, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf7, (50265, 768), (768, 1), 0), reinterpret_tensor(buf8, (50265, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_169 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    sub = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_2 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_12 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_6 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_2 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_14 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_9 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_3 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_6 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_30 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_14 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_5 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_6 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_36 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_10 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_48 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_22 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_8 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_50 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_52 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_12 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_9 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_54 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_14 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_66 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_15 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_30 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_11 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_70 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_33 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_12 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_72 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_18 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_84 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_19 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_38 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_20 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_22 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_102 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_23 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_46 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_26 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_120 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_54 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_122 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_124 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_28 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_30 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_138 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_62 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_142 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_65 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_144 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_34 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_156 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_35 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_70 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_158 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_160 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_162 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_38 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_174 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_39 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_78 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_29 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_180 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_42 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_192 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_86 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_32 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_44 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_89 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_46 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_210 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_94 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_35 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_212 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_36 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_115 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_101 = rand_strided((512, 50265), (50265, 1), device='cpu', dtype=torch.float32)
    convert_element_type_49 = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_147 = rand_strided((50265, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_163 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_168 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_169 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_178 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_180 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_193 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_48 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_196 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_203 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_205 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_209 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_53 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_220 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_221 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_230 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_243 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_58 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_246 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_253 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_259 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_263 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_268 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_269 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_63 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_271 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_280 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_293 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_294 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_68 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_295 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_296 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_303 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_305 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_309 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_313 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_318 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_319 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_73 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_320 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_321 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_328 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_330 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_334 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_338 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_343 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_78 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_345 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_346 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_353 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_359 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_363 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_369 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_83 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_370 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_371 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_378 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_380 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_384 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_394 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_88 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_395 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_396 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_403 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_405 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_409 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_413 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_418 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_419 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_93 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_420 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_421 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_428 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_430 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_443 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_444 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_98 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_445 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_446 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_453 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 50265), (25735680, 50265, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_163, primals_168, primals_169, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, addmm_36, mul_115, view_218, sub_101, convert_element_type_49, permute_147, div_51, permute_151, permute_155, permute_159, permute_163, permute_168, permute_169, alias_43, permute_170, permute_171, permute_178, permute_180, permute_184, permute_188, permute_193, permute_194, alias_48, permute_195, permute_196, permute_203, permute_205, permute_209, permute_213, permute_218, permute_219, alias_53, permute_220, permute_221, permute_228, permute_230, permute_234, permute_238, permute_243, permute_244, alias_58, permute_245, permute_246, permute_253, permute_255, permute_259, permute_263, permute_268, permute_269, alias_63, permute_270, permute_271, permute_278, permute_280, permute_284, permute_288, permute_293, permute_294, alias_68, permute_295, permute_296, permute_303, permute_305, permute_309, permute_313, permute_318, permute_319, alias_73, permute_320, permute_321, permute_328, permute_330, permute_334, permute_338, permute_343, permute_344, alias_78, permute_345, permute_346, permute_353, permute_355, permute_359, permute_363, permute_368, permute_369, alias_83, permute_370, permute_371, permute_378, permute_380, permute_384, permute_388, permute_393, permute_394, alias_88, permute_395, permute_396, permute_403, permute_405, permute_409, permute_413, permute_418, permute_419, alias_93, permute_420, permute_421, permute_428, permute_430, permute_434, permute_438, permute_443, permute_444, alias_98, permute_445, permute_446, permute_453, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForMaskedLM', benchmark_compiled_module)
