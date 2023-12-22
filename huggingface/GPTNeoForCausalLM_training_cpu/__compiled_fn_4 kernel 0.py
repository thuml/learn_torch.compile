
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6382632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(6382632L); x0<static_cast<long>(6382639L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = static_cast<float>(0.0);
                out_ptr0[static_cast<long>(x0)] = tmp0;
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(127L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(127L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (50257L*x0)));
                    auto tmp1 = c10::convert<int>(x0);
                    auto tmp2 = static_cast<int>(127);
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
                    auto tmp2 = static_cast<long>(127);
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


cpp_fused_native_layer_norm_backward_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp11 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = static_cast<float>(2048.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 - tmp13;
                    auto tmp15 = at::vec::Vectorized<float>(tmp0);
                    auto tmp16 = tmp15 * tmp14;
                    tmp16.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_92 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_108 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_116 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_123 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_124 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_131 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_132 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_139 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_140 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_147 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_148 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_155 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_156 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_162 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_163 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_164 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_170 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_171 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_172 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_175 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_176 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_177 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_179 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_180 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_181 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_182 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_183 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_184 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_185 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_186 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_187 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_188 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8192L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(2048.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_189 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_nll_loss_forward_where_190 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_191 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_192 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (16384L*x0)), static_cast<long>(128L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (16384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (128L*x0) + (2048L*x1) + (2048L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_193 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_194 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const long* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp23 = in_ptr6[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    auto tmp24 = static_cast<int>(-1);
                    auto tmp25 = tmp23 == tmp24;
                    auto tmp26 = static_cast<float>(0.0);
                    auto tmp27 = to_float_mask(tmp25);
                    auto tmp28 = at::vec::Vectorized<float>(tmp26);
                    auto tmp29 = decltype(tmp28)::blendv(tmp22, tmp28, tmp27);
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    tmp29.store(out_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(1L))
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


cpp_fused_embedding_dense_backward_195 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(102926336L); x0+=static_cast<long>(8L))
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
    primals_3, primals_10, primals_16, primals_23, primals_29, primals_36, primals_42, primals_49, primals_55, primals_62, primals_68, primals_75, primals_81, primals_88, primals_94, primals_101, primals_107, primals_114, primals_120, primals_127, primals_133, primals_140, primals_146, primals_153, primals_159, primals_166, primals_172, primals_179, primals_185, primals_192, primals_198, primals_205, primals_211, primals_218, primals_224, primals_231, primals_237, primals_244, primals_250, primals_257, primals_263, primals_270, primals_276, primals_283, primals_289, primals_296, primals_302, primals_309, primals_315, primals_343, view, view_1, mul, view_2, slice_4, view_18, mul_2, view_20, addmm_1, tanh, view_22, mul_8, view_24, slice_8, view_40, mul_10, view_42, addmm_4, tanh_1, view_44, mul_16, view_46, slice_12, view_62, mul_18, view_64, addmm_7, tanh_2, view_66, mul_24, view_68, slice_16, view_84, mul_26, view_86, addmm_10, tanh_3, view_88, mul_32, view_90, slice_20, view_106, mul_34, view_108, addmm_13, tanh_4, view_110, mul_40, view_112, slice_24, view_128, mul_42, view_130, addmm_16, tanh_5, view_132, mul_48, view_134, slice_28, view_150, mul_50, view_152, addmm_19, tanh_6, view_154, mul_56, view_156, slice_32, view_172, mul_58, view_174, addmm_22, tanh_7, view_176, mul_64, view_178, slice_36, view_194, mul_66, view_196, addmm_25, tanh_8, view_198, mul_72, view_200, slice_40, view_216, mul_74, view_218, addmm_28, tanh_9, view_220, mul_80, view_222, slice_44, view_238, mul_82, view_240, addmm_31, tanh_10, view_242, mul_88, view_244, slice_48, view_260, mul_90, view_262, addmm_34, tanh_11, view_264, mul_96, view_266, slice_52, view_282, mul_98, view_284, addmm_37, tanh_12, view_286, mul_104, view_288, slice_56, view_304, mul_106, view_306, addmm_40, tanh_13, view_308, mul_112, view_310, slice_60, view_326, mul_114, view_328, addmm_43, tanh_14, view_330, mul_120, view_332, slice_64, view_348, mul_122, view_350, addmm_46, tanh_15, view_352, mul_128, view_354, slice_68, view_370, mul_130, view_372, addmm_49, tanh_16, view_374, mul_136, view_376, slice_72, view_392, mul_138, view_394, addmm_52, tanh_17, view_396, mul_144, view_398, slice_76, view_414, mul_146, view_416, addmm_55, tanh_18, view_418, mul_152, view_420, slice_80, view_436, mul_154, view_438, addmm_58, tanh_19, view_440, mul_160, view_442, slice_84, view_458, mul_162, view_460, addmm_61, tanh_20, view_462, mul_168, view_464, slice_88, view_480, mul_170, view_482, addmm_64, tanh_21, view_484, mul_176, view_486, slice_92, view_502, mul_178, view_504, addmm_67, tanh_22, view_506, mul_184, view_508, slice_96, view_524, mul_186, view_526, addmm_70, tanh_23, view_528, mul_192, view_531, sub_74, convert_element_type, permute_267, div_26, permute_269, permute_273, div_27, permute_277, permute_282, permute_283, alias_51, permute_284, permute_285, permute_292, permute_296, permute_300, div_28, permute_302, permute_306, div_29, permute_310, permute_315, permute_316, alias_53, permute_317, permute_318, permute_325, permute_329, permute_333, div_30, permute_335, permute_339, div_31, permute_343, permute_348, permute_349, alias_55, permute_350, permute_351, permute_358, permute_362, permute_366, div_32, permute_368, permute_372, div_33, permute_376, permute_381, permute_382, alias_57, permute_383, permute_384, permute_391, permute_395, permute_399, div_34, permute_401, permute_405, div_35, permute_409, permute_414, permute_415, alias_59, permute_416, permute_417, permute_424, permute_428, permute_432, div_36, permute_434, permute_438, div_37, permute_442, permute_447, permute_448, alias_61, permute_449, permute_450, permute_457, permute_461, permute_465, div_38, permute_467, permute_471, div_39, permute_475, permute_480, permute_481, alias_63, permute_482, permute_483, permute_490, permute_494, permute_498, div_40, permute_500, permute_504, div_41, permute_508, permute_513, permute_514, alias_65, permute_515, permute_516, permute_523, permute_527, permute_531, div_42, permute_533, permute_537, div_43, permute_541, permute_546, permute_547, alias_67, permute_548, permute_549, permute_556, permute_560, permute_564, div_44, permute_566, permute_570, div_45, permute_574, permute_579, permute_580, alias_69, permute_581, permute_582, permute_589, permute_593, permute_597, div_46, permute_599, permute_603, div_47, permute_607, permute_612, permute_613, alias_71, permute_614, permute_615, permute_622, permute_626, permute_630, div_48, permute_632, permute_636, div_49, permute_640, permute_645, permute_646, alias_73, permute_647, permute_648, permute_655, permute_659, permute_663, div_50, permute_665, permute_669, div_51, permute_673, permute_678, permute_679, alias_75, permute_680, permute_681, permute_688, permute_692, permute_696, div_52, permute_698, permute_702, div_53, permute_706, permute_711, permute_712, alias_77, permute_713, permute_714, permute_721, permute_725, permute_729, div_54, permute_731, permute_735, div_55, permute_739, permute_744, permute_745, alias_79, permute_746, permute_747, permute_754, permute_758, permute_762, div_56, permute_764, permute_768, div_57, permute_772, permute_777, permute_778, alias_81, permute_779, permute_780, permute_787, permute_791, permute_795, div_58, permute_797, permute_801, div_59, permute_805, permute_810, permute_811, alias_83, permute_812, permute_813, permute_820, permute_824, permute_828, div_60, permute_830, permute_834, div_61, permute_838, permute_843, permute_844, alias_85, permute_845, permute_846, permute_853, permute_857, permute_861, div_62, permute_863, permute_867, div_63, permute_871, permute_876, permute_877, alias_87, permute_878, permute_879, permute_886, permute_890, permute_894, div_64, permute_896, permute_900, div_65, permute_904, permute_909, permute_910, alias_89, permute_911, permute_912, permute_919, permute_923, permute_927, div_66, permute_929, permute_933, div_67, permute_937, permute_942, permute_943, alias_91, permute_944, permute_945, permute_952, permute_956, permute_960, div_68, permute_962, permute_966, div_69, permute_970, permute_975, permute_976, alias_93, permute_977, permute_978, permute_985, permute_989, permute_993, div_70, permute_995, permute_999, div_71, permute_1003, permute_1008, permute_1009, alias_95, permute_1010, permute_1011, permute_1018, permute_1022, permute_1026, div_72, permute_1028, permute_1032, div_73, permute_1036, permute_1041, permute_1042, alias_97, permute_1043, permute_1044, permute_1051, permute_1055, permute_1059, div_74, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50 = args
    args.clear()
    assert_size_stride(primals_3, (2048, ), (1, ))
    assert_size_stride(primals_10, (2048, ), (1, ))
    assert_size_stride(primals_16, (2048, ), (1, ))
    assert_size_stride(primals_23, (2048, ), (1, ))
    assert_size_stride(primals_29, (2048, ), (1, ))
    assert_size_stride(primals_36, (2048, ), (1, ))
    assert_size_stride(primals_42, (2048, ), (1, ))
    assert_size_stride(primals_49, (2048, ), (1, ))
    assert_size_stride(primals_55, (2048, ), (1, ))
    assert_size_stride(primals_62, (2048, ), (1, ))
    assert_size_stride(primals_68, (2048, ), (1, ))
    assert_size_stride(primals_75, (2048, ), (1, ))
    assert_size_stride(primals_81, (2048, ), (1, ))
    assert_size_stride(primals_88, (2048, ), (1, ))
    assert_size_stride(primals_94, (2048, ), (1, ))
    assert_size_stride(primals_101, (2048, ), (1, ))
    assert_size_stride(primals_107, (2048, ), (1, ))
    assert_size_stride(primals_114, (2048, ), (1, ))
    assert_size_stride(primals_120, (2048, ), (1, ))
    assert_size_stride(primals_127, (2048, ), (1, ))
    assert_size_stride(primals_133, (2048, ), (1, ))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_146, (2048, ), (1, ))
    assert_size_stride(primals_153, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_166, (2048, ), (1, ))
    assert_size_stride(primals_172, (2048, ), (1, ))
    assert_size_stride(primals_179, (2048, ), (1, ))
    assert_size_stride(primals_185, (2048, ), (1, ))
    assert_size_stride(primals_192, (2048, ), (1, ))
    assert_size_stride(primals_198, (2048, ), (1, ))
    assert_size_stride(primals_205, (2048, ), (1, ))
    assert_size_stride(primals_211, (2048, ), (1, ))
    assert_size_stride(primals_218, (2048, ), (1, ))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_231, (2048, ), (1, ))
    assert_size_stride(primals_237, (2048, ), (1, ))
    assert_size_stride(primals_244, (2048, ), (1, ))
    assert_size_stride(primals_250, (2048, ), (1, ))
    assert_size_stride(primals_257, (2048, ), (1, ))
    assert_size_stride(primals_263, (2048, ), (1, ))
    assert_size_stride(primals_270, (2048, ), (1, ))
    assert_size_stride(primals_276, (2048, ), (1, ))
    assert_size_stride(primals_283, (2048, ), (1, ))
    assert_size_stride(primals_289, (2048, ), (1, ))
    assert_size_stride(primals_296, (2048, ), (1, ))
    assert_size_stride(primals_302, (2048, ), (1, ))
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_315, (2048, ), (1, ))
    assert_size_stride(primals_343, (1, 128), (128, 1))
    assert_size_stride(view, (1, 128), (128, 1))
    assert_size_stride(view_1, (1, 128), (128, 1))
    assert_size_stride(mul, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_2, (128, 2048), (2048, 1))
    assert_size_stride(slice_4, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_18, (128, 2048), (2048, 1))
    assert_size_stride(mul_2, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_20, (128, 2048), (2048, 1))
    assert_size_stride(addmm_1, (128, 8192), (8192, 1))
    assert_size_stride(tanh, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_22, (128, 8192), (8192, 1))
    assert_size_stride(mul_8, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_24, (128, 2048), (2048, 1))
    assert_size_stride(slice_8, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_40, (128, 2048), (2048, 1))
    assert_size_stride(mul_10, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_42, (128, 2048), (2048, 1))
    assert_size_stride(addmm_4, (128, 8192), (8192, 1))
    assert_size_stride(tanh_1, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_44, (128, 8192), (8192, 1))
    assert_size_stride(mul_16, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_46, (128, 2048), (2048, 1))
    assert_size_stride(slice_12, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_62, (128, 2048), (2048, 1))
    assert_size_stride(mul_18, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_64, (128, 2048), (2048, 1))
    assert_size_stride(addmm_7, (128, 8192), (8192, 1))
    assert_size_stride(tanh_2, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_66, (128, 8192), (8192, 1))
    assert_size_stride(mul_24, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_68, (128, 2048), (2048, 1))
    assert_size_stride(slice_16, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_84, (128, 2048), (2048, 1))
    assert_size_stride(mul_26, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_86, (128, 2048), (2048, 1))
    assert_size_stride(addmm_10, (128, 8192), (8192, 1))
    assert_size_stride(tanh_3, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_88, (128, 8192), (8192, 1))
    assert_size_stride(mul_32, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_90, (128, 2048), (2048, 1))
    assert_size_stride(slice_20, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_106, (128, 2048), (2048, 1))
    assert_size_stride(mul_34, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_108, (128, 2048), (2048, 1))
    assert_size_stride(addmm_13, (128, 8192), (8192, 1))
    assert_size_stride(tanh_4, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_110, (128, 8192), (8192, 1))
    assert_size_stride(mul_40, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_112, (128, 2048), (2048, 1))
    assert_size_stride(slice_24, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_128, (128, 2048), (2048, 1))
    assert_size_stride(mul_42, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_130, (128, 2048), (2048, 1))
    assert_size_stride(addmm_16, (128, 8192), (8192, 1))
    assert_size_stride(tanh_5, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_132, (128, 8192), (8192, 1))
    assert_size_stride(mul_48, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_134, (128, 2048), (2048, 1))
    assert_size_stride(slice_28, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_150, (128, 2048), (2048, 1))
    assert_size_stride(mul_50, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_152, (128, 2048), (2048, 1))
    assert_size_stride(addmm_19, (128, 8192), (8192, 1))
    assert_size_stride(tanh_6, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_154, (128, 8192), (8192, 1))
    assert_size_stride(mul_56, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_156, (128, 2048), (2048, 1))
    assert_size_stride(slice_32, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_172, (128, 2048), (2048, 1))
    assert_size_stride(mul_58, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_174, (128, 2048), (2048, 1))
    assert_size_stride(addmm_22, (128, 8192), (8192, 1))
    assert_size_stride(tanh_7, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_176, (128, 8192), (8192, 1))
    assert_size_stride(mul_64, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_178, (128, 2048), (2048, 1))
    assert_size_stride(slice_36, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_194, (128, 2048), (2048, 1))
    assert_size_stride(mul_66, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_196, (128, 2048), (2048, 1))
    assert_size_stride(addmm_25, (128, 8192), (8192, 1))
    assert_size_stride(tanh_8, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_198, (128, 8192), (8192, 1))
    assert_size_stride(mul_72, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_200, (128, 2048), (2048, 1))
    assert_size_stride(slice_40, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_216, (128, 2048), (2048, 1))
    assert_size_stride(mul_74, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_218, (128, 2048), (2048, 1))
    assert_size_stride(addmm_28, (128, 8192), (8192, 1))
    assert_size_stride(tanh_9, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_220, (128, 8192), (8192, 1))
    assert_size_stride(mul_80, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_222, (128, 2048), (2048, 1))
    assert_size_stride(slice_44, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_238, (128, 2048), (2048, 1))
    assert_size_stride(mul_82, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_240, (128, 2048), (2048, 1))
    assert_size_stride(addmm_31, (128, 8192), (8192, 1))
    assert_size_stride(tanh_10, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_242, (128, 8192), (8192, 1))
    assert_size_stride(mul_88, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_244, (128, 2048), (2048, 1))
    assert_size_stride(slice_48, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_260, (128, 2048), (2048, 1))
    assert_size_stride(mul_90, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_262, (128, 2048), (2048, 1))
    assert_size_stride(addmm_34, (128, 8192), (8192, 1))
    assert_size_stride(tanh_11, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_264, (128, 8192), (8192, 1))
    assert_size_stride(mul_96, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_266, (128, 2048), (2048, 1))
    assert_size_stride(slice_52, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_282, (128, 2048), (2048, 1))
    assert_size_stride(mul_98, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_284, (128, 2048), (2048, 1))
    assert_size_stride(addmm_37, (128, 8192), (8192, 1))
    assert_size_stride(tanh_12, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_286, (128, 8192), (8192, 1))
    assert_size_stride(mul_104, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_288, (128, 2048), (2048, 1))
    assert_size_stride(slice_56, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_304, (128, 2048), (2048, 1))
    assert_size_stride(mul_106, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_306, (128, 2048), (2048, 1))
    assert_size_stride(addmm_40, (128, 8192), (8192, 1))
    assert_size_stride(tanh_13, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_308, (128, 8192), (8192, 1))
    assert_size_stride(mul_112, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_310, (128, 2048), (2048, 1))
    assert_size_stride(slice_60, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_326, (128, 2048), (2048, 1))
    assert_size_stride(mul_114, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_328, (128, 2048), (2048, 1))
    assert_size_stride(addmm_43, (128, 8192), (8192, 1))
    assert_size_stride(tanh_14, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_330, (128, 8192), (8192, 1))
    assert_size_stride(mul_120, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_332, (128, 2048), (2048, 1))
    assert_size_stride(slice_64, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_348, (128, 2048), (2048, 1))
    assert_size_stride(mul_122, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_350, (128, 2048), (2048, 1))
    assert_size_stride(addmm_46, (128, 8192), (8192, 1))
    assert_size_stride(tanh_15, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_352, (128, 8192), (8192, 1))
    assert_size_stride(mul_128, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_354, (128, 2048), (2048, 1))
    assert_size_stride(slice_68, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_370, (128, 2048), (2048, 1))
    assert_size_stride(mul_130, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_372, (128, 2048), (2048, 1))
    assert_size_stride(addmm_49, (128, 8192), (8192, 1))
    assert_size_stride(tanh_16, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_374, (128, 8192), (8192, 1))
    assert_size_stride(mul_136, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_376, (128, 2048), (2048, 1))
    assert_size_stride(slice_72, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_392, (128, 2048), (2048, 1))
    assert_size_stride(mul_138, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_394, (128, 2048), (2048, 1))
    assert_size_stride(addmm_52, (128, 8192), (8192, 1))
    assert_size_stride(tanh_17, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_396, (128, 8192), (8192, 1))
    assert_size_stride(mul_144, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_398, (128, 2048), (2048, 1))
    assert_size_stride(slice_76, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_414, (128, 2048), (2048, 1))
    assert_size_stride(mul_146, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_416, (128, 2048), (2048, 1))
    assert_size_stride(addmm_55, (128, 8192), (8192, 1))
    assert_size_stride(tanh_18, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_418, (128, 8192), (8192, 1))
    assert_size_stride(mul_152, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_420, (128, 2048), (2048, 1))
    assert_size_stride(slice_80, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_436, (128, 2048), (2048, 1))
    assert_size_stride(mul_154, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_438, (128, 2048), (2048, 1))
    assert_size_stride(addmm_58, (128, 8192), (8192, 1))
    assert_size_stride(tanh_19, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_440, (128, 8192), (8192, 1))
    assert_size_stride(mul_160, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_442, (128, 2048), (2048, 1))
    assert_size_stride(slice_84, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_458, (128, 2048), (2048, 1))
    assert_size_stride(mul_162, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_460, (128, 2048), (2048, 1))
    assert_size_stride(addmm_61, (128, 8192), (8192, 1))
    assert_size_stride(tanh_20, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_462, (128, 8192), (8192, 1))
    assert_size_stride(mul_168, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_464, (128, 2048), (2048, 1))
    assert_size_stride(slice_88, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_480, (128, 2048), (2048, 1))
    assert_size_stride(mul_170, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_482, (128, 2048), (2048, 1))
    assert_size_stride(addmm_64, (128, 8192), (8192, 1))
    assert_size_stride(tanh_21, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_484, (128, 8192), (8192, 1))
    assert_size_stride(mul_176, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_486, (128, 2048), (2048, 1))
    assert_size_stride(slice_92, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_502, (128, 2048), (2048, 1))
    assert_size_stride(mul_178, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_504, (128, 2048), (2048, 1))
    assert_size_stride(addmm_67, (128, 8192), (8192, 1))
    assert_size_stride(tanh_22, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_506, (128, 8192), (8192, 1))
    assert_size_stride(mul_184, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_508, (128, 2048), (2048, 1))
    assert_size_stride(slice_96, (1, 1, 128, 128), (4194304, 4194304, 2048, 1))
    assert_size_stride(view_524, (128, 2048), (2048, 1))
    assert_size_stride(mul_186, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_526, (128, 2048), (2048, 1))
    assert_size_stride(addmm_70, (128, 8192), (8192, 1))
    assert_size_stride(tanh_23, (1, 128, 8192), (1048576, 8192, 1))
    assert_size_stride(view_528, (128, 8192), (8192, 1))
    assert_size_stride(mul_192, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(view_531, (128, 2048), (2048, 1))
    assert_size_stride(sub_74, (127, 50257), (50257, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_267, (50257, 2048), (2048, 1))
    assert_size_stride(div_26, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_269, (2048, 8192), (8192, 1))
    assert_size_stride(permute_273, (8192, 2048), (2048, 1))
    assert_size_stride(div_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_277, (2048, 2048), (2048, 1))
    assert_size_stride(permute_282, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_283, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_51, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_284, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_285, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_292, (2048, 2048), (2048, 1))
    assert_size_stride(permute_296, (2048, 2048), (2048, 1))
    assert_size_stride(permute_300, (2048, 2048), (2048, 1))
    assert_size_stride(div_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_302, (2048, 8192), (8192, 1))
    assert_size_stride(permute_306, (8192, 2048), (2048, 1))
    assert_size_stride(div_29, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_310, (2048, 2048), (2048, 1))
    assert_size_stride(permute_315, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_316, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_53, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_317, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_318, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_325, (2048, 2048), (2048, 1))
    assert_size_stride(permute_329, (2048, 2048), (2048, 1))
    assert_size_stride(permute_333, (2048, 2048), (2048, 1))
    assert_size_stride(div_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_335, (2048, 8192), (8192, 1))
    assert_size_stride(permute_339, (8192, 2048), (2048, 1))
    assert_size_stride(div_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_343, (2048, 2048), (2048, 1))
    assert_size_stride(permute_348, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_349, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_55, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_350, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_351, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_358, (2048, 2048), (2048, 1))
    assert_size_stride(permute_362, (2048, 2048), (2048, 1))
    assert_size_stride(permute_366, (2048, 2048), (2048, 1))
    assert_size_stride(div_32, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_368, (2048, 8192), (8192, 1))
    assert_size_stride(permute_372, (8192, 2048), (2048, 1))
    assert_size_stride(div_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_376, (2048, 2048), (2048, 1))
    assert_size_stride(permute_381, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_382, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_57, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_383, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_384, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_391, (2048, 2048), (2048, 1))
    assert_size_stride(permute_395, (2048, 2048), (2048, 1))
    assert_size_stride(permute_399, (2048, 2048), (2048, 1))
    assert_size_stride(div_34, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_401, (2048, 8192), (8192, 1))
    assert_size_stride(permute_405, (8192, 2048), (2048, 1))
    assert_size_stride(div_35, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_409, (2048, 2048), (2048, 1))
    assert_size_stride(permute_414, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_415, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_59, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_416, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_417, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_424, (2048, 2048), (2048, 1))
    assert_size_stride(permute_428, (2048, 2048), (2048, 1))
    assert_size_stride(permute_432, (2048, 2048), (2048, 1))
    assert_size_stride(div_36, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_434, (2048, 8192), (8192, 1))
    assert_size_stride(permute_438, (8192, 2048), (2048, 1))
    assert_size_stride(div_37, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_442, (2048, 2048), (2048, 1))
    assert_size_stride(permute_447, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_448, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_61, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_449, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_450, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_457, (2048, 2048), (2048, 1))
    assert_size_stride(permute_461, (2048, 2048), (2048, 1))
    assert_size_stride(permute_465, (2048, 2048), (2048, 1))
    assert_size_stride(div_38, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_467, (2048, 8192), (8192, 1))
    assert_size_stride(permute_471, (8192, 2048), (2048, 1))
    assert_size_stride(div_39, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_475, (2048, 2048), (2048, 1))
    assert_size_stride(permute_480, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_481, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_63, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_482, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_483, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_490, (2048, 2048), (2048, 1))
    assert_size_stride(permute_494, (2048, 2048), (2048, 1))
    assert_size_stride(permute_498, (2048, 2048), (2048, 1))
    assert_size_stride(div_40, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_500, (2048, 8192), (8192, 1))
    assert_size_stride(permute_504, (8192, 2048), (2048, 1))
    assert_size_stride(div_41, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_508, (2048, 2048), (2048, 1))
    assert_size_stride(permute_513, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_514, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_65, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_515, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_516, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_523, (2048, 2048), (2048, 1))
    assert_size_stride(permute_527, (2048, 2048), (2048, 1))
    assert_size_stride(permute_531, (2048, 2048), (2048, 1))
    assert_size_stride(div_42, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_533, (2048, 8192), (8192, 1))
    assert_size_stride(permute_537, (8192, 2048), (2048, 1))
    assert_size_stride(div_43, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_541, (2048, 2048), (2048, 1))
    assert_size_stride(permute_546, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_547, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_67, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_548, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_549, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_556, (2048, 2048), (2048, 1))
    assert_size_stride(permute_560, (2048, 2048), (2048, 1))
    assert_size_stride(permute_564, (2048, 2048), (2048, 1))
    assert_size_stride(div_44, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_566, (2048, 8192), (8192, 1))
    assert_size_stride(permute_570, (8192, 2048), (2048, 1))
    assert_size_stride(div_45, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_574, (2048, 2048), (2048, 1))
    assert_size_stride(permute_579, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_580, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_69, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_581, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_582, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_589, (2048, 2048), (2048, 1))
    assert_size_stride(permute_593, (2048, 2048), (2048, 1))
    assert_size_stride(permute_597, (2048, 2048), (2048, 1))
    assert_size_stride(div_46, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_599, (2048, 8192), (8192, 1))
    assert_size_stride(permute_603, (8192, 2048), (2048, 1))
    assert_size_stride(div_47, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_607, (2048, 2048), (2048, 1))
    assert_size_stride(permute_612, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_613, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_71, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_614, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_615, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_622, (2048, 2048), (2048, 1))
    assert_size_stride(permute_626, (2048, 2048), (2048, 1))
    assert_size_stride(permute_630, (2048, 2048), (2048, 1))
    assert_size_stride(div_48, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_632, (2048, 8192), (8192, 1))
    assert_size_stride(permute_636, (8192, 2048), (2048, 1))
    assert_size_stride(div_49, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_640, (2048, 2048), (2048, 1))
    assert_size_stride(permute_645, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_646, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_73, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_647, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_648, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_655, (2048, 2048), (2048, 1))
    assert_size_stride(permute_659, (2048, 2048), (2048, 1))
    assert_size_stride(permute_663, (2048, 2048), (2048, 1))
    assert_size_stride(div_50, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_665, (2048, 8192), (8192, 1))
    assert_size_stride(permute_669, (8192, 2048), (2048, 1))
    assert_size_stride(div_51, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_673, (2048, 2048), (2048, 1))
    assert_size_stride(permute_678, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_679, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_75, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_680, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_681, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_688, (2048, 2048), (2048, 1))
    assert_size_stride(permute_692, (2048, 2048), (2048, 1))
    assert_size_stride(permute_696, (2048, 2048), (2048, 1))
    assert_size_stride(div_52, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_698, (2048, 8192), (8192, 1))
    assert_size_stride(permute_702, (8192, 2048), (2048, 1))
    assert_size_stride(div_53, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_706, (2048, 2048), (2048, 1))
    assert_size_stride(permute_711, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_712, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_77, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_713, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_714, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_721, (2048, 2048), (2048, 1))
    assert_size_stride(permute_725, (2048, 2048), (2048, 1))
    assert_size_stride(permute_729, (2048, 2048), (2048, 1))
    assert_size_stride(div_54, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_731, (2048, 8192), (8192, 1))
    assert_size_stride(permute_735, (8192, 2048), (2048, 1))
    assert_size_stride(div_55, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_739, (2048, 2048), (2048, 1))
    assert_size_stride(permute_744, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_745, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_79, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_746, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_747, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_754, (2048, 2048), (2048, 1))
    assert_size_stride(permute_758, (2048, 2048), (2048, 1))
    assert_size_stride(permute_762, (2048, 2048), (2048, 1))
    assert_size_stride(div_56, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_764, (2048, 8192), (8192, 1))
    assert_size_stride(permute_768, (8192, 2048), (2048, 1))
    assert_size_stride(div_57, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_772, (2048, 2048), (2048, 1))
    assert_size_stride(permute_777, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_778, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_81, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_779, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_780, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_787, (2048, 2048), (2048, 1))
    assert_size_stride(permute_791, (2048, 2048), (2048, 1))
    assert_size_stride(permute_795, (2048, 2048), (2048, 1))
    assert_size_stride(div_58, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_797, (2048, 8192), (8192, 1))
    assert_size_stride(permute_801, (8192, 2048), (2048, 1))
    assert_size_stride(div_59, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_805, (2048, 2048), (2048, 1))
    assert_size_stride(permute_810, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_811, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_83, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_812, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_813, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_820, (2048, 2048), (2048, 1))
    assert_size_stride(permute_824, (2048, 2048), (2048, 1))
    assert_size_stride(permute_828, (2048, 2048), (2048, 1))
    assert_size_stride(div_60, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_830, (2048, 8192), (8192, 1))
    assert_size_stride(permute_834, (8192, 2048), (2048, 1))
    assert_size_stride(div_61, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_838, (2048, 2048), (2048, 1))
    assert_size_stride(permute_843, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_844, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_85, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_845, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_846, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_853, (2048, 2048), (2048, 1))
    assert_size_stride(permute_857, (2048, 2048), (2048, 1))
    assert_size_stride(permute_861, (2048, 2048), (2048, 1))
    assert_size_stride(div_62, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_863, (2048, 8192), (8192, 1))
    assert_size_stride(permute_867, (8192, 2048), (2048, 1))
    assert_size_stride(div_63, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_871, (2048, 2048), (2048, 1))
    assert_size_stride(permute_876, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_877, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_87, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_878, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_879, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_886, (2048, 2048), (2048, 1))
    assert_size_stride(permute_890, (2048, 2048), (2048, 1))
    assert_size_stride(permute_894, (2048, 2048), (2048, 1))
    assert_size_stride(div_64, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_896, (2048, 8192), (8192, 1))
    assert_size_stride(permute_900, (8192, 2048), (2048, 1))
    assert_size_stride(div_65, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_904, (2048, 2048), (2048, 1))
    assert_size_stride(permute_909, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_910, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_89, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_911, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_912, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_919, (2048, 2048), (2048, 1))
    assert_size_stride(permute_923, (2048, 2048), (2048, 1))
    assert_size_stride(permute_927, (2048, 2048), (2048, 1))
    assert_size_stride(div_66, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_929, (2048, 8192), (8192, 1))
    assert_size_stride(permute_933, (8192, 2048), (2048, 1))
    assert_size_stride(div_67, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_937, (2048, 2048), (2048, 1))
    assert_size_stride(permute_942, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_943, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_91, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_944, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_945, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_952, (2048, 2048), (2048, 1))
    assert_size_stride(permute_956, (2048, 2048), (2048, 1))
    assert_size_stride(permute_960, (2048, 2048), (2048, 1))
    assert_size_stride(div_68, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_962, (2048, 8192), (8192, 1))
    assert_size_stride(permute_966, (8192, 2048), (2048, 1))
    assert_size_stride(div_69, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_970, (2048, 2048), (2048, 1))
    assert_size_stride(permute_975, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_976, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_93, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_977, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_978, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_985, (2048, 2048), (2048, 1))
    assert_size_stride(permute_989, (2048, 2048), (2048, 1))
    assert_size_stride(permute_993, (2048, 2048), (2048, 1))
    assert_size_stride(div_70, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_995, (2048, 8192), (8192, 1))
    assert_size_stride(permute_999, (8192, 2048), (2048, 1))
    assert_size_stride(div_71, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1003, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1008, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1009, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_95, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_1010, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_1011, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_1018, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1022, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1026, (2048, 2048), (2048, 1))
    assert_size_stride(div_72, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1028, (2048, 8192), (8192, 1))
    assert_size_stride(permute_1032, (8192, 2048), (2048, 1))
    assert_size_stride(div_73, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_1036, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1041, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1042, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(alias_97, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(permute_1043, (16, 128, 128), (128, 1, 2048))
    assert_size_stride(permute_1044, (16, 128, 128), (128, 2048, 1))
    assert_size_stride(permute_1051, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1055, (2048, 2048), (2048, 1))
    assert_size_stride(permute_1059, (2048, 2048), (2048, 1))
    assert_size_stride(div_74, (1, 128, 1), (128, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 50257), (6432896, 50257, 1))
    assert_size_stride(tangents_3, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_4, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_5, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_6, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_7, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_8, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_9, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_10, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_11, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_12, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_13, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_14, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_15, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_16, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_17, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_18, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_19, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_20, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_21, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_22, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_23, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_24, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_25, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_26, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_27, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_28, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_29, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_30, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_31, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_32, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_33, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_34, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_35, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_36, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_37, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_38, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_39, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_40, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_41, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_42, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_43, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_44, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_45, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_46, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_47, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_48, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_49, (1, 16, 128, 128), (262144, 16384, 128, 1))
    assert_size_stride(tangents_50, (1, 16, 128, 128), (262144, 16384, 128, 1))
    buf0 = empty((127, 50257), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_343.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 128, 50257), device='cpu', dtype=torch.float32)
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_slice_backward_1(c_void_p(buf0.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_74.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del buf0
    del buf4
    del convert_element_type
    del primals_343
    del sub_74
    del tangents_1
    del tangents_2
    buf6 = empty((50257, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (50257, 128), (1, 50257), 0), view_531, out=buf6)
    del view_531
    buf7 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (128, 50257), (50257, 1), 0), permute_267, out=buf7)
    del buf5
    del permute_267
    buf8 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf10 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf11 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf12 = empty((2048, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_2(c_void_p(buf7.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(mul_192.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del div_26
    del mul_192
    del primals_315
    buf13 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf10, (128, 2048), (2048, 1), 0), permute_269, out=buf13)
    del permute_269
    buf14 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf10, (2048, 128), (1, 2048), 0), view_528, out=buf14)
    del view_528
    buf15 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf13, (1, 128, 8192), (1048576, 8192, 1), 0); del buf13  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_3(c_void_p(buf16.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(tanh_23.data_ptr()), c_void_p(buf15.data_ptr()))
    del addmm_70
    del tanh_23
    buf17 = buf7; del buf7  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (128, 8192), (8192, 1), 0), permute_273, out=buf17)
    del permute_273
    buf18 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (8192, 128), (1, 8192), 0), view_526, out=buf18)
    del view_526
    buf19 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf20 = buf9; del buf9  # reuse
    buf21 = buf8; del buf8  # reuse
    buf22 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf23 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf24 = buf10; del buf10  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_4(c_void_p(buf24.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(mul_186.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del div_27
    del mul_186
    del primals_309
    buf25 = buf17; del buf17  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (128, 2048), (2048, 1), 0), permute_277, out=buf25)
    del permute_277
    buf26 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (2048, 128), (1, 2048), 0), view_524, out=buf26)
    del view_524
    buf27 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_5(c_void_p(buf24.data_ptr()), c_void_p(buf27.data_ptr()))
    buf28 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_282, reinterpret_tensor(buf25, (16, 128, 128), (128, 2048, 1), 0), out=buf28)
    del permute_282
    buf29 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (16, 128, 128), (128, 2048, 1), 0), permute_283, out=buf29)
    del permute_283
    buf30 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf31 = reinterpret_tensor(buf29, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf29  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_6(c_void_p(buf31.data_ptr()), c_void_p(alias_51.data_ptr()), c_void_p(slice_96.data_ptr()), c_void_p(buf30.data_ptr()))
    del alias_51
    del slice_96
    buf32 = reinterpret_tensor(buf25, (16, 128, 128), (16384, 128, 1), 0); del buf25  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_284, reinterpret_tensor(buf31, (16, 128, 128), (16384, 128, 1), 0), out=buf32)
    del permute_284
    buf33 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf31, (16, 128, 128), (16384, 128, 1), 0), permute_285, out=buf33)
    del permute_285
    buf34 = reinterpret_tensor(buf31, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf31  # reuse
    cpp_fused_clone_7(c_void_p(tangents_50.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf34.data_ptr()))
    del tangents_50
    buf35 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (2048, 128), (1, 2048), 0), view_508, out=buf35)
    buf36 = reinterpret_tensor(buf28, (128, 2048), (2048, 1), 0); del buf28  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (128, 2048), (2048, 1), 0), permute_292, out=buf36)
    del permute_292
    buf37 = buf34; del buf34  # reuse
    cpp_fused_clone_8(c_void_p(tangents_49.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf37.data_ptr()))
    del tangents_49
    buf38 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (2048, 128), (1, 2048), 0), view_508, out=buf38)
    buf39 = reinterpret_tensor(buf32, (128, 2048), (2048, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (128, 2048), (2048, 1), 0), permute_296, out=buf39)
    del permute_296
    buf40 = reinterpret_tensor(buf37, (128, 2048), (2048, 1), 0); del buf37  # reuse
    cpp_fused_view_9(c_void_p(buf33.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (2048, 128), (1, 2048), 0), view_508, out=buf41)
    del view_508
    buf42 = reinterpret_tensor(buf33, (128, 2048), (2048, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, permute_300, out=buf42)
    del permute_300
    buf43 = buf21; del buf21  # reuse
    buf44 = buf20; del buf20  # reuse
    buf45 = reinterpret_tensor(buf30, (2048, ), (1, ), 0); del buf30  # reuse
    buf46 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf47 = buf24; del buf24  # reuse
    cpp_fused_add_native_layer_norm_backward_10(c_void_p(buf47.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(mul_184.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del div_28
    del mul_184
    del primals_302
    buf48 = reinterpret_tensor(buf16, (128, 8192), (8192, 1), 0); del buf16  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (128, 2048), (2048, 1), 0), permute_302, out=buf48)
    del permute_302
    buf49 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (2048, 128), (1, 2048), 0), view_506, out=buf49)
    del view_506
    buf50 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf51 = reinterpret_tensor(buf48, (1, 128, 8192), (1048576, 8192, 1), 0); del buf48  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_11(c_void_p(buf51.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(addmm_67.data_ptr()), c_void_p(tanh_22.data_ptr()), c_void_p(buf50.data_ptr()))
    del addmm_67
    del tanh_22
    buf52 = buf42; del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (128, 8192), (8192, 1), 0), permute_306, out=buf52)
    del permute_306
    buf53 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (8192, 128), (1, 8192), 0), view_504, out=buf53)
    del view_504
    buf54 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf55 = buf44; del buf44  # reuse
    buf56 = buf43; del buf43  # reuse
    buf57 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf58 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf59 = buf47; del buf47  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_12(c_void_p(buf59.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(mul_178.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del div_29
    del mul_178
    del primals_296
    buf60 = buf52; del buf52  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf59, (128, 2048), (2048, 1), 0), permute_310, out=buf60)
    del permute_310
    buf61 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf59, (2048, 128), (1, 2048), 0), view_502, out=buf61)
    del view_502
    buf62 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_13(c_void_p(buf59.data_ptr()), c_void_p(buf62.data_ptr()))
    buf63 = reinterpret_tensor(buf39, (16, 128, 128), (16384, 128, 1), 0); del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_315, reinterpret_tensor(buf60, (16, 128, 128), (128, 2048, 1), 0), out=buf63)
    del permute_315
    buf64 = reinterpret_tensor(buf36, (16, 128, 128), (16384, 128, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf60, (16, 128, 128), (128, 2048, 1), 0), permute_316, out=buf64)
    del permute_316
    buf65 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf66 = reinterpret_tensor(buf64, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf64  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_14(c_void_p(buf66.data_ptr()), c_void_p(alias_53.data_ptr()), c_void_p(slice_92.data_ptr()), c_void_p(buf65.data_ptr()))
    del alias_53
    del slice_92
    buf67 = reinterpret_tensor(buf60, (16, 128, 128), (16384, 128, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_317, reinterpret_tensor(buf66, (16, 128, 128), (16384, 128, 1), 0), out=buf67)
    del permute_317
    buf68 = reinterpret_tensor(buf40, (16, 128, 128), (16384, 128, 1), 0); del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf66, (16, 128, 128), (16384, 128, 1), 0), permute_318, out=buf68)
    del permute_318
    buf69 = reinterpret_tensor(buf66, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf66  # reuse
    cpp_fused_clone_15(c_void_p(tangents_48.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf69.data_ptr()))
    del tangents_48
    buf70 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf69, (2048, 128), (1, 2048), 0), view_486, out=buf70)
    buf71 = reinterpret_tensor(buf63, (128, 2048), (2048, 1), 0); del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf69, (128, 2048), (2048, 1), 0), permute_325, out=buf71)
    del permute_325
    buf72 = buf69; del buf69  # reuse
    cpp_fused_clone_16(c_void_p(tangents_47.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf72.data_ptr()))
    del tangents_47
    buf73 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (2048, 128), (1, 2048), 0), view_486, out=buf73)
    buf74 = reinterpret_tensor(buf67, (128, 2048), (2048, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (128, 2048), (2048, 1), 0), permute_329, out=buf74)
    del permute_329
    buf75 = reinterpret_tensor(buf72, (128, 2048), (2048, 1), 0); del buf72  # reuse
    cpp_fused_view_17(c_void_p(buf68.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (2048, 128), (1, 2048), 0), view_486, out=buf76)
    del view_486
    buf77 = reinterpret_tensor(buf68, (128, 2048), (2048, 1), 0); del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf75, permute_333, out=buf77)
    del permute_333
    buf78 = buf56; del buf56  # reuse
    buf79 = buf55; del buf55  # reuse
    buf80 = reinterpret_tensor(buf65, (2048, ), (1, ), 0); del buf65  # reuse
    buf81 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf82 = buf59; del buf59  # reuse
    cpp_fused_add_native_layer_norm_backward_18(c_void_p(buf82.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(mul_176.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del div_30
    del mul_176
    del primals_289
    buf83 = reinterpret_tensor(buf51, (128, 8192), (8192, 1), 0); del buf51  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (128, 2048), (2048, 1), 0), permute_335, out=buf83)
    del permute_335
    buf84 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (2048, 128), (1, 2048), 0), view_484, out=buf84)
    del view_484
    buf85 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf83, (1, 128, 8192), (1048576, 8192, 1), 0); del buf83  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_19(c_void_p(buf86.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(addmm_64.data_ptr()), c_void_p(tanh_21.data_ptr()), c_void_p(buf85.data_ptr()))
    del addmm_64
    del tanh_21
    buf87 = buf77; del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (128, 8192), (8192, 1), 0), permute_339, out=buf87)
    del permute_339
    buf88 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (8192, 128), (1, 8192), 0), view_482, out=buf88)
    del view_482
    buf89 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf90 = buf79; del buf79  # reuse
    buf91 = buf78; del buf78  # reuse
    buf92 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf93 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf94 = buf82; del buf82  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_20(c_void_p(buf94.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(mul_170.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    del div_31
    del mul_170
    del primals_283
    buf95 = buf87; del buf87  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (128, 2048), (2048, 1), 0), permute_343, out=buf95)
    del permute_343
    buf96 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (2048, 128), (1, 2048), 0), view_480, out=buf96)
    del view_480
    buf97 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_21(c_void_p(buf94.data_ptr()), c_void_p(buf97.data_ptr()))
    buf98 = reinterpret_tensor(buf74, (16, 128, 128), (16384, 128, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_348, reinterpret_tensor(buf95, (16, 128, 128), (128, 2048, 1), 0), out=buf98)
    del permute_348
    buf99 = reinterpret_tensor(buf71, (16, 128, 128), (16384, 128, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (16, 128, 128), (128, 2048, 1), 0), permute_349, out=buf99)
    del permute_349
    buf100 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf101 = reinterpret_tensor(buf99, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf99  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_22(c_void_p(buf101.data_ptr()), c_void_p(alias_55.data_ptr()), c_void_p(slice_88.data_ptr()), c_void_p(buf100.data_ptr()))
    del alias_55
    del slice_88
    buf102 = reinterpret_tensor(buf95, (16, 128, 128), (16384, 128, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_350, reinterpret_tensor(buf101, (16, 128, 128), (16384, 128, 1), 0), out=buf102)
    del permute_350
    buf103 = reinterpret_tensor(buf75, (16, 128, 128), (16384, 128, 1), 0); del buf75  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf101, (16, 128, 128), (16384, 128, 1), 0), permute_351, out=buf103)
    del permute_351
    buf104 = reinterpret_tensor(buf101, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf101  # reuse
    cpp_fused_clone_23(c_void_p(tangents_46.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf104.data_ptr()))
    del tangents_46
    buf105 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (2048, 128), (1, 2048), 0), view_464, out=buf105)
    buf106 = reinterpret_tensor(buf98, (128, 2048), (2048, 1), 0); del buf98  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (128, 2048), (2048, 1), 0), permute_358, out=buf106)
    del permute_358
    buf107 = buf104; del buf104  # reuse
    cpp_fused_clone_24(c_void_p(tangents_45.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf107.data_ptr()))
    del tangents_45
    buf108 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (2048, 128), (1, 2048), 0), view_464, out=buf108)
    buf109 = reinterpret_tensor(buf102, (128, 2048), (2048, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (128, 2048), (2048, 1), 0), permute_362, out=buf109)
    del permute_362
    buf110 = reinterpret_tensor(buf107, (128, 2048), (2048, 1), 0); del buf107  # reuse
    cpp_fused_view_25(c_void_p(buf103.data_ptr()), c_void_p(buf110.data_ptr()))
    buf111 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (2048, 128), (1, 2048), 0), view_464, out=buf111)
    del view_464
    buf112 = reinterpret_tensor(buf103, (128, 2048), (2048, 1), 0); del buf103  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf110, permute_366, out=buf112)
    del permute_366
    buf113 = buf91; del buf91  # reuse
    buf114 = buf90; del buf90  # reuse
    buf115 = reinterpret_tensor(buf100, (2048, ), (1, ), 0); del buf100  # reuse
    buf116 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf117 = reinterpret_tensor(buf106, (1, 128, 2048), (262144, 2048, 1), 0); del buf106  # reuse
    cpp_fused_add_native_layer_norm_backward_26(c_void_p(buf117.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(mul_168.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del div_32
    del mul_168
    del primals_276
    buf118 = reinterpret_tensor(buf86, (128, 8192), (8192, 1), 0); del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (128, 2048), (2048, 1), 0), permute_368, out=buf118)
    del permute_368
    buf119 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (2048, 128), (1, 2048), 0), view_462, out=buf119)
    del view_462
    buf120 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf121 = reinterpret_tensor(buf118, (1, 128, 8192), (1048576, 8192, 1), 0); del buf118  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_27(c_void_p(buf121.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(addmm_61.data_ptr()), c_void_p(tanh_20.data_ptr()), c_void_p(buf120.data_ptr()))
    del addmm_61
    del tanh_20
    buf122 = reinterpret_tensor(buf94, (128, 2048), (2048, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (128, 8192), (8192, 1), 0), permute_372, out=buf122)
    del permute_372
    buf123 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (8192, 128), (1, 8192), 0), view_460, out=buf123)
    del view_460
    buf124 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf125 = buf114; del buf114  # reuse
    buf126 = buf113; del buf113  # reuse
    buf127 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf128 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf129 = buf117; del buf117  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_28(c_void_p(buf129.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(mul_162.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del div_33
    del mul_162
    del primals_270
    buf130 = buf122; del buf122  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (128, 2048), (2048, 1), 0), permute_376, out=buf130)
    del permute_376
    buf131 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (2048, 128), (1, 2048), 0), view_458, out=buf131)
    del view_458
    buf132 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_29(c_void_p(buf129.data_ptr()), c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf112, (16, 128, 128), (16384, 128, 1), 0); del buf112  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_381, reinterpret_tensor(buf130, (16, 128, 128), (128, 2048, 1), 0), out=buf133)
    del permute_381
    buf134 = reinterpret_tensor(buf109, (16, 128, 128), (16384, 128, 1), 0); del buf109  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf130, (16, 128, 128), (128, 2048, 1), 0), permute_382, out=buf134)
    del permute_382
    buf135 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf136 = reinterpret_tensor(buf134, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf134  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_30(c_void_p(buf136.data_ptr()), c_void_p(alias_57.data_ptr()), c_void_p(slice_84.data_ptr()), c_void_p(buf135.data_ptr()))
    del alias_57
    del slice_84
    buf137 = reinterpret_tensor(buf130, (16, 128, 128), (16384, 128, 1), 0); del buf130  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_383, reinterpret_tensor(buf136, (16, 128, 128), (16384, 128, 1), 0), out=buf137)
    del permute_383
    buf138 = reinterpret_tensor(buf110, (16, 128, 128), (16384, 128, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (16, 128, 128), (16384, 128, 1), 0), permute_384, out=buf138)
    del permute_384
    buf139 = reinterpret_tensor(buf136, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf136  # reuse
    cpp_fused_clone_31(c_void_p(tangents_44.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf139.data_ptr()))
    del tangents_44
    buf140 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (2048, 128), (1, 2048), 0), view_442, out=buf140)
    buf141 = reinterpret_tensor(buf133, (128, 2048), (2048, 1), 0); del buf133  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (128, 2048), (2048, 1), 0), permute_391, out=buf141)
    del permute_391
    buf142 = buf139; del buf139  # reuse
    cpp_fused_clone_32(c_void_p(tangents_43.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf142.data_ptr()))
    del tangents_43
    buf143 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (2048, 128), (1, 2048), 0), view_442, out=buf143)
    buf144 = reinterpret_tensor(buf137, (128, 2048), (2048, 1), 0); del buf137  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (128, 2048), (2048, 1), 0), permute_395, out=buf144)
    del permute_395
    buf145 = reinterpret_tensor(buf142, (128, 2048), (2048, 1), 0); del buf142  # reuse
    cpp_fused_view_33(c_void_p(buf138.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (2048, 128), (1, 2048), 0), view_442, out=buf146)
    del view_442
    buf147 = reinterpret_tensor(buf138, (128, 2048), (2048, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf145, permute_399, out=buf147)
    del permute_399
    buf148 = buf126; del buf126  # reuse
    buf149 = buf125; del buf125  # reuse
    buf150 = reinterpret_tensor(buf135, (2048, ), (1, ), 0); del buf135  # reuse
    buf151 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf152 = buf129; del buf129  # reuse
    cpp_fused_add_native_layer_norm_backward_34(c_void_p(buf152.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(mul_160.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del div_34
    del mul_160
    del primals_263
    buf153 = reinterpret_tensor(buf121, (128, 8192), (8192, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (128, 2048), (2048, 1), 0), permute_401, out=buf153)
    del permute_401
    buf154 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (2048, 128), (1, 2048), 0), view_440, out=buf154)
    del view_440
    buf155 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf156 = reinterpret_tensor(buf153, (1, 128, 8192), (1048576, 8192, 1), 0); del buf153  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_35(c_void_p(buf156.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(tanh_19.data_ptr()), c_void_p(buf155.data_ptr()))
    del addmm_58
    del tanh_19
    buf157 = buf147; del buf147  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf156, (128, 8192), (8192, 1), 0), permute_405, out=buf157)
    del permute_405
    buf158 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf156, (8192, 128), (1, 8192), 0), view_438, out=buf158)
    del view_438
    buf159 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf160 = buf149; del buf149  # reuse
    buf161 = buf148; del buf148  # reuse
    buf162 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf163 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf164 = buf152; del buf152  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_36(c_void_p(buf164.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(mul_154.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del div_35
    del mul_154
    del primals_257
    buf165 = buf157; del buf157  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf164, (128, 2048), (2048, 1), 0), permute_409, out=buf165)
    del permute_409
    buf166 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf164, (2048, 128), (1, 2048), 0), view_436, out=buf166)
    del view_436
    buf167 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_37(c_void_p(buf164.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = reinterpret_tensor(buf144, (16, 128, 128), (16384, 128, 1), 0); del buf144  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_414, reinterpret_tensor(buf165, (16, 128, 128), (128, 2048, 1), 0), out=buf168)
    del permute_414
    buf169 = reinterpret_tensor(buf141, (16, 128, 128), (16384, 128, 1), 0); del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf165, (16, 128, 128), (128, 2048, 1), 0), permute_415, out=buf169)
    del permute_415
    buf170 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf169, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf169  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_38(c_void_p(buf171.data_ptr()), c_void_p(alias_59.data_ptr()), c_void_p(slice_80.data_ptr()), c_void_p(buf170.data_ptr()))
    del alias_59
    del slice_80
    buf172 = reinterpret_tensor(buf165, (16, 128, 128), (16384, 128, 1), 0); del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_416, reinterpret_tensor(buf171, (16, 128, 128), (16384, 128, 1), 0), out=buf172)
    del permute_416
    buf173 = reinterpret_tensor(buf145, (16, 128, 128), (16384, 128, 1), 0); del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf171, (16, 128, 128), (16384, 128, 1), 0), permute_417, out=buf173)
    del permute_417
    buf174 = reinterpret_tensor(buf171, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf171  # reuse
    cpp_fused_clone_39(c_void_p(tangents_42.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf174.data_ptr()))
    del tangents_42
    buf175 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (2048, 128), (1, 2048), 0), view_420, out=buf175)
    buf176 = reinterpret_tensor(buf168, (128, 2048), (2048, 1), 0); del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 2048), (2048, 1), 0), permute_424, out=buf176)
    del permute_424
    buf177 = buf174; del buf174  # reuse
    cpp_fused_clone_40(c_void_p(tangents_41.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf177.data_ptr()))
    del tangents_41
    buf178 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (2048, 128), (1, 2048), 0), view_420, out=buf178)
    buf179 = reinterpret_tensor(buf172, (128, 2048), (2048, 1), 0); del buf172  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (128, 2048), (2048, 1), 0), permute_428, out=buf179)
    del permute_428
    buf180 = reinterpret_tensor(buf177, (128, 2048), (2048, 1), 0); del buf177  # reuse
    cpp_fused_view_41(c_void_p(buf173.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (2048, 128), (1, 2048), 0), view_420, out=buf181)
    del view_420
    buf182 = reinterpret_tensor(buf173, (128, 2048), (2048, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf180, permute_432, out=buf182)
    del permute_432
    buf183 = buf161; del buf161  # reuse
    buf184 = buf160; del buf160  # reuse
    buf185 = reinterpret_tensor(buf170, (2048, ), (1, ), 0); del buf170  # reuse
    buf186 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf187 = buf164; del buf164  # reuse
    cpp_fused_add_native_layer_norm_backward_42(c_void_p(buf187.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(mul_152.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del div_36
    del mul_152
    del primals_250
    buf188 = reinterpret_tensor(buf156, (128, 8192), (8192, 1), 0); del buf156  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (128, 2048), (2048, 1), 0), permute_434, out=buf188)
    del permute_434
    buf189 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (2048, 128), (1, 2048), 0), view_418, out=buf189)
    del view_418
    buf190 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf191 = reinterpret_tensor(buf188, (1, 128, 8192), (1048576, 8192, 1), 0); del buf188  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_43(c_void_p(buf191.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(addmm_55.data_ptr()), c_void_p(tanh_18.data_ptr()), c_void_p(buf190.data_ptr()))
    del addmm_55
    del tanh_18
    buf192 = buf182; del buf182  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf191, (128, 8192), (8192, 1), 0), permute_438, out=buf192)
    del permute_438
    buf193 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf191, (8192, 128), (1, 8192), 0), view_416, out=buf193)
    del view_416
    buf194 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf195 = buf184; del buf184  # reuse
    buf196 = buf183; del buf183  # reuse
    buf197 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf198 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf199 = buf187; del buf187  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_44(c_void_p(buf199.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(mul_146.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    del div_37
    del mul_146
    del primals_244
    buf200 = buf192; del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (128, 2048), (2048, 1), 0), permute_442, out=buf200)
    del permute_442
    buf201 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (2048, 128), (1, 2048), 0), view_414, out=buf201)
    del view_414
    buf202 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_45(c_void_p(buf199.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf179, (16, 128, 128), (16384, 128, 1), 0); del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_447, reinterpret_tensor(buf200, (16, 128, 128), (128, 2048, 1), 0), out=buf203)
    del permute_447
    buf204 = reinterpret_tensor(buf176, (16, 128, 128), (16384, 128, 1), 0); del buf176  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf200, (16, 128, 128), (128, 2048, 1), 0), permute_448, out=buf204)
    del permute_448
    buf205 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf206 = reinterpret_tensor(buf204, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf204  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_46(c_void_p(buf206.data_ptr()), c_void_p(alias_61.data_ptr()), c_void_p(slice_76.data_ptr()), c_void_p(buf205.data_ptr()))
    del alias_61
    del slice_76
    buf207 = reinterpret_tensor(buf200, (16, 128, 128), (16384, 128, 1), 0); del buf200  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_449, reinterpret_tensor(buf206, (16, 128, 128), (16384, 128, 1), 0), out=buf207)
    del permute_449
    buf208 = reinterpret_tensor(buf180, (16, 128, 128), (16384, 128, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf206, (16, 128, 128), (16384, 128, 1), 0), permute_450, out=buf208)
    del permute_450
    buf209 = reinterpret_tensor(buf206, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf206  # reuse
    cpp_fused_clone_47(c_void_p(tangents_40.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf209.data_ptr()))
    del tangents_40
    buf210 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (2048, 128), (1, 2048), 0), view_398, out=buf210)
    buf211 = reinterpret_tensor(buf203, (128, 2048), (2048, 1), 0); del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (128, 2048), (2048, 1), 0), permute_457, out=buf211)
    del permute_457
    buf212 = buf209; del buf209  # reuse
    cpp_fused_clone_48(c_void_p(tangents_39.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf212.data_ptr()))
    del tangents_39
    buf213 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (2048, 128), (1, 2048), 0), view_398, out=buf213)
    buf214 = reinterpret_tensor(buf207, (128, 2048), (2048, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (128, 2048), (2048, 1), 0), permute_461, out=buf214)
    del permute_461
    buf215 = reinterpret_tensor(buf212, (128, 2048), (2048, 1), 0); del buf212  # reuse
    cpp_fused_view_49(c_void_p(buf208.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (2048, 128), (1, 2048), 0), view_398, out=buf216)
    del view_398
    buf217 = reinterpret_tensor(buf208, (128, 2048), (2048, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf215, permute_465, out=buf217)
    del permute_465
    buf218 = buf196; del buf196  # reuse
    buf219 = buf195; del buf195  # reuse
    buf220 = reinterpret_tensor(buf205, (2048, ), (1, ), 0); del buf205  # reuse
    buf221 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf222 = buf199; del buf199  # reuse
    cpp_fused_add_native_layer_norm_backward_50(c_void_p(buf222.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(mul_144.data_ptr()), c_void_p(div_38.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    del div_38
    del mul_144
    del primals_237
    buf223 = reinterpret_tensor(buf191, (128, 8192), (8192, 1), 0); del buf191  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (128, 2048), (2048, 1), 0), permute_467, out=buf223)
    del permute_467
    buf224 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (2048, 128), (1, 2048), 0), view_396, out=buf224)
    del view_396
    buf225 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf226 = reinterpret_tensor(buf223, (1, 128, 8192), (1048576, 8192, 1), 0); del buf223  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_51(c_void_p(buf226.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(tanh_17.data_ptr()), c_void_p(buf225.data_ptr()))
    del addmm_52
    del tanh_17
    buf227 = buf217; del buf217  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (128, 8192), (8192, 1), 0), permute_471, out=buf227)
    del permute_471
    buf228 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (8192, 128), (1, 8192), 0), view_394, out=buf228)
    del view_394
    buf229 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf230 = buf219; del buf219  # reuse
    buf231 = buf218; del buf218  # reuse
    buf232 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf233 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf234 = buf222; del buf222  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_52(c_void_p(buf234.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(mul_138.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    del div_39
    del mul_138
    del primals_231
    buf235 = buf227; del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf234, (128, 2048), (2048, 1), 0), permute_475, out=buf235)
    del permute_475
    buf236 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf234, (2048, 128), (1, 2048), 0), view_392, out=buf236)
    del view_392
    buf237 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_53(c_void_p(buf234.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = reinterpret_tensor(buf214, (16, 128, 128), (16384, 128, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_480, reinterpret_tensor(buf235, (16, 128, 128), (128, 2048, 1), 0), out=buf238)
    del permute_480
    buf239 = reinterpret_tensor(buf211, (16, 128, 128), (16384, 128, 1), 0); del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf235, (16, 128, 128), (128, 2048, 1), 0), permute_481, out=buf239)
    del permute_481
    buf240 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf241 = reinterpret_tensor(buf239, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf239  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_54(c_void_p(buf241.data_ptr()), c_void_p(alias_63.data_ptr()), c_void_p(slice_72.data_ptr()), c_void_p(buf240.data_ptr()))
    del alias_63
    del slice_72
    buf242 = reinterpret_tensor(buf235, (16, 128, 128), (16384, 128, 1), 0); del buf235  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_482, reinterpret_tensor(buf241, (16, 128, 128), (16384, 128, 1), 0), out=buf242)
    del permute_482
    buf243 = reinterpret_tensor(buf215, (16, 128, 128), (16384, 128, 1), 0); del buf215  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf241, (16, 128, 128), (16384, 128, 1), 0), permute_483, out=buf243)
    del permute_483
    buf244 = reinterpret_tensor(buf241, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf241  # reuse
    cpp_fused_clone_55(c_void_p(tangents_38.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf244.data_ptr()))
    del tangents_38
    buf245 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (2048, 128), (1, 2048), 0), view_376, out=buf245)
    buf246 = reinterpret_tensor(buf238, (128, 2048), (2048, 1), 0); del buf238  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (128, 2048), (2048, 1), 0), permute_490, out=buf246)
    del permute_490
    buf247 = buf244; del buf244  # reuse
    cpp_fused_clone_56(c_void_p(tangents_37.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf247.data_ptr()))
    del tangents_37
    buf248 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (2048, 128), (1, 2048), 0), view_376, out=buf248)
    buf249 = reinterpret_tensor(buf242, (128, 2048), (2048, 1), 0); del buf242  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (128, 2048), (2048, 1), 0), permute_494, out=buf249)
    del permute_494
    buf250 = reinterpret_tensor(buf247, (128, 2048), (2048, 1), 0); del buf247  # reuse
    cpp_fused_view_57(c_void_p(buf243.data_ptr()), c_void_p(buf250.data_ptr()))
    buf251 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf250, (2048, 128), (1, 2048), 0), view_376, out=buf251)
    del view_376
    buf252 = reinterpret_tensor(buf243, (128, 2048), (2048, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf250, permute_498, out=buf252)
    del permute_498
    buf253 = buf231; del buf231  # reuse
    buf254 = buf230; del buf230  # reuse
    buf255 = reinterpret_tensor(buf240, (2048, ), (1, ), 0); del buf240  # reuse
    buf256 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf257 = buf234; del buf234  # reuse
    cpp_fused_add_native_layer_norm_backward_58(c_void_p(buf257.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(mul_136.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del div_40
    del mul_136
    del primals_224
    buf258 = reinterpret_tensor(buf226, (128, 8192), (8192, 1), 0); del buf226  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (128, 2048), (2048, 1), 0), permute_500, out=buf258)
    del permute_500
    buf259 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (2048, 128), (1, 2048), 0), view_374, out=buf259)
    del view_374
    buf260 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf261 = reinterpret_tensor(buf258, (1, 128, 8192), (1048576, 8192, 1), 0); del buf258  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_59(c_void_p(buf261.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(addmm_49.data_ptr()), c_void_p(tanh_16.data_ptr()), c_void_p(buf260.data_ptr()))
    del addmm_49
    del tanh_16
    buf262 = buf252; del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (128, 8192), (8192, 1), 0), permute_504, out=buf262)
    del permute_504
    buf263 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (8192, 128), (1, 8192), 0), view_372, out=buf263)
    del view_372
    buf264 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf265 = buf254; del buf254  # reuse
    buf266 = buf253; del buf253  # reuse
    buf267 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf268 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf269 = buf257; del buf257  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_60(c_void_p(buf269.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(mul_130.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    del div_41
    del mul_130
    del primals_218
    buf270 = buf262; del buf262  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (128, 2048), (2048, 1), 0), permute_508, out=buf270)
    del permute_508
    buf271 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (2048, 128), (1, 2048), 0), view_370, out=buf271)
    del view_370
    buf272 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_61(c_void_p(buf269.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = reinterpret_tensor(buf249, (16, 128, 128), (16384, 128, 1), 0); del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_513, reinterpret_tensor(buf270, (16, 128, 128), (128, 2048, 1), 0), out=buf273)
    del permute_513
    buf274 = reinterpret_tensor(buf246, (16, 128, 128), (16384, 128, 1), 0); del buf246  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf270, (16, 128, 128), (128, 2048, 1), 0), permute_514, out=buf274)
    del permute_514
    buf275 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf276 = reinterpret_tensor(buf274, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf274  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_62(c_void_p(buf276.data_ptr()), c_void_p(alias_65.data_ptr()), c_void_p(slice_68.data_ptr()), c_void_p(buf275.data_ptr()))
    del alias_65
    del slice_68
    buf277 = reinterpret_tensor(buf270, (16, 128, 128), (16384, 128, 1), 0); del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_515, reinterpret_tensor(buf276, (16, 128, 128), (16384, 128, 1), 0), out=buf277)
    del permute_515
    buf278 = reinterpret_tensor(buf250, (16, 128, 128), (16384, 128, 1), 0); del buf250  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf276, (16, 128, 128), (16384, 128, 1), 0), permute_516, out=buf278)
    del permute_516
    buf279 = reinterpret_tensor(buf276, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf276  # reuse
    cpp_fused_clone_63(c_void_p(tangents_36.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf279.data_ptr()))
    del tangents_36
    buf280 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (2048, 128), (1, 2048), 0), view_354, out=buf280)
    buf281 = reinterpret_tensor(buf273, (128, 2048), (2048, 1), 0); del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (128, 2048), (2048, 1), 0), permute_523, out=buf281)
    del permute_523
    buf282 = buf279; del buf279  # reuse
    cpp_fused_clone_64(c_void_p(tangents_35.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf282.data_ptr()))
    del tangents_35
    buf283 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (2048, 128), (1, 2048), 0), view_354, out=buf283)
    buf284 = reinterpret_tensor(buf277, (128, 2048), (2048, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (128, 2048), (2048, 1), 0), permute_527, out=buf284)
    del permute_527
    buf285 = reinterpret_tensor(buf282, (128, 2048), (2048, 1), 0); del buf282  # reuse
    cpp_fused_view_65(c_void_p(buf278.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf285, (2048, 128), (1, 2048), 0), view_354, out=buf286)
    del view_354
    buf287 = reinterpret_tensor(buf278, (128, 2048), (2048, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf285, permute_531, out=buf287)
    del permute_531
    buf288 = buf266; del buf266  # reuse
    buf289 = buf265; del buf265  # reuse
    buf290 = reinterpret_tensor(buf275, (2048, ), (1, ), 0); del buf275  # reuse
    buf291 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf292 = buf269; del buf269  # reuse
    cpp_fused_add_native_layer_norm_backward_66(c_void_p(buf292.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(mul_128.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del div_42
    del mul_128
    del primals_211
    buf293 = reinterpret_tensor(buf261, (128, 8192), (8192, 1), 0); del buf261  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (128, 2048), (2048, 1), 0), permute_533, out=buf293)
    del permute_533
    buf294 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (2048, 128), (1, 2048), 0), view_352, out=buf294)
    del view_352
    buf295 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf296 = reinterpret_tensor(buf293, (1, 128, 8192), (1048576, 8192, 1), 0); del buf293  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_67(c_void_p(buf296.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(tanh_15.data_ptr()), c_void_p(buf295.data_ptr()))
    del addmm_46
    del tanh_15
    buf297 = buf287; del buf287  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (128, 8192), (8192, 1), 0), permute_537, out=buf297)
    del permute_537
    buf298 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (8192, 128), (1, 8192), 0), view_350, out=buf298)
    del view_350
    buf299 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf300 = buf289; del buf289  # reuse
    buf301 = buf288; del buf288  # reuse
    buf302 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf303 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf304 = buf292; del buf292  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_68(c_void_p(buf304.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(mul_122.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()))
    del div_43
    del mul_122
    del primals_205
    buf305 = buf297; del buf297  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (128, 2048), (2048, 1), 0), permute_541, out=buf305)
    del permute_541
    buf306 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (2048, 128), (1, 2048), 0), view_348, out=buf306)
    del view_348
    buf307 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_69(c_void_p(buf304.data_ptr()), c_void_p(buf307.data_ptr()))
    buf308 = reinterpret_tensor(buf284, (16, 128, 128), (16384, 128, 1), 0); del buf284  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_546, reinterpret_tensor(buf305, (16, 128, 128), (128, 2048, 1), 0), out=buf308)
    del permute_546
    buf309 = reinterpret_tensor(buf281, (16, 128, 128), (16384, 128, 1), 0); del buf281  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf305, (16, 128, 128), (128, 2048, 1), 0), permute_547, out=buf309)
    del permute_547
    buf310 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf311 = reinterpret_tensor(buf309, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf309  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_70(c_void_p(buf311.data_ptr()), c_void_p(alias_67.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf310.data_ptr()))
    del alias_67
    del slice_64
    buf312 = reinterpret_tensor(buf305, (16, 128, 128), (16384, 128, 1), 0); del buf305  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_548, reinterpret_tensor(buf311, (16, 128, 128), (16384, 128, 1), 0), out=buf312)
    del permute_548
    buf313 = reinterpret_tensor(buf285, (16, 128, 128), (16384, 128, 1), 0); del buf285  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf311, (16, 128, 128), (16384, 128, 1), 0), permute_549, out=buf313)
    del permute_549
    buf314 = reinterpret_tensor(buf311, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf311  # reuse
    cpp_fused_clone_71(c_void_p(tangents_34.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf314.data_ptr()))
    del tangents_34
    buf315 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf314, (2048, 128), (1, 2048), 0), view_332, out=buf315)
    buf316 = reinterpret_tensor(buf308, (128, 2048), (2048, 1), 0); del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf314, (128, 2048), (2048, 1), 0), permute_556, out=buf316)
    del permute_556
    buf317 = buf314; del buf314  # reuse
    cpp_fused_clone_72(c_void_p(tangents_33.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf317.data_ptr()))
    del tangents_33
    buf318 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (2048, 128), (1, 2048), 0), view_332, out=buf318)
    buf319 = reinterpret_tensor(buf312, (128, 2048), (2048, 1), 0); del buf312  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (128, 2048), (2048, 1), 0), permute_560, out=buf319)
    del permute_560
    buf320 = reinterpret_tensor(buf317, (128, 2048), (2048, 1), 0); del buf317  # reuse
    cpp_fused_view_73(c_void_p(buf313.data_ptr()), c_void_p(buf320.data_ptr()))
    buf321 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf320, (2048, 128), (1, 2048), 0), view_332, out=buf321)
    del view_332
    buf322 = reinterpret_tensor(buf313, (128, 2048), (2048, 1), 0); del buf313  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, permute_564, out=buf322)
    del permute_564
    buf323 = buf301; del buf301  # reuse
    buf324 = buf300; del buf300  # reuse
    buf325 = reinterpret_tensor(buf310, (2048, ), (1, ), 0); del buf310  # reuse
    buf326 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf327 = buf304; del buf304  # reuse
    cpp_fused_add_native_layer_norm_backward_74(c_void_p(buf327.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(mul_120.data_ptr()), c_void_p(div_44.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del div_44
    del mul_120
    del primals_198
    buf328 = reinterpret_tensor(buf296, (128, 8192), (8192, 1), 0); del buf296  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (128, 2048), (2048, 1), 0), permute_566, out=buf328)
    del permute_566
    buf329 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (2048, 128), (1, 2048), 0), view_330, out=buf329)
    del view_330
    buf330 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf331 = reinterpret_tensor(buf328, (1, 128, 8192), (1048576, 8192, 1), 0); del buf328  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_75(c_void_p(buf331.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(addmm_43.data_ptr()), c_void_p(tanh_14.data_ptr()), c_void_p(buf330.data_ptr()))
    del addmm_43
    del tanh_14
    buf332 = buf322; del buf322  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (128, 8192), (8192, 1), 0), permute_570, out=buf332)
    del permute_570
    buf333 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (8192, 128), (1, 8192), 0), view_328, out=buf333)
    del view_328
    buf334 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf335 = buf324; del buf324  # reuse
    buf336 = buf323; del buf323  # reuse
    buf337 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf338 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf339 = buf327; del buf327  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_76(c_void_p(buf339.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(mul_114.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    del div_45
    del mul_114
    del primals_192
    buf340 = buf332; del buf332  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf339, (128, 2048), (2048, 1), 0), permute_574, out=buf340)
    del permute_574
    buf341 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf339, (2048, 128), (1, 2048), 0), view_326, out=buf341)
    del view_326
    buf342 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_77(c_void_p(buf339.data_ptr()), c_void_p(buf342.data_ptr()))
    buf343 = reinterpret_tensor(buf319, (16, 128, 128), (16384, 128, 1), 0); del buf319  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_579, reinterpret_tensor(buf340, (16, 128, 128), (128, 2048, 1), 0), out=buf343)
    del permute_579
    buf344 = reinterpret_tensor(buf316, (16, 128, 128), (16384, 128, 1), 0); del buf316  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf340, (16, 128, 128), (128, 2048, 1), 0), permute_580, out=buf344)
    del permute_580
    buf345 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf346 = reinterpret_tensor(buf344, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf344  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_78(c_void_p(buf346.data_ptr()), c_void_p(alias_69.data_ptr()), c_void_p(slice_60.data_ptr()), c_void_p(buf345.data_ptr()))
    del alias_69
    del slice_60
    buf347 = reinterpret_tensor(buf340, (16, 128, 128), (16384, 128, 1), 0); del buf340  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_581, reinterpret_tensor(buf346, (16, 128, 128), (16384, 128, 1), 0), out=buf347)
    del permute_581
    buf348 = reinterpret_tensor(buf320, (16, 128, 128), (16384, 128, 1), 0); del buf320  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf346, (16, 128, 128), (16384, 128, 1), 0), permute_582, out=buf348)
    del permute_582
    buf349 = reinterpret_tensor(buf346, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf346  # reuse
    cpp_fused_clone_79(c_void_p(tangents_32.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf349.data_ptr()))
    del tangents_32
    buf350 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (2048, 128), (1, 2048), 0), view_310, out=buf350)
    buf351 = reinterpret_tensor(buf343, (128, 2048), (2048, 1), 0); del buf343  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (128, 2048), (2048, 1), 0), permute_589, out=buf351)
    del permute_589
    buf352 = buf349; del buf349  # reuse
    cpp_fused_clone_80(c_void_p(tangents_31.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf352.data_ptr()))
    del tangents_31
    buf353 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (2048, 128), (1, 2048), 0), view_310, out=buf353)
    buf354 = reinterpret_tensor(buf347, (128, 2048), (2048, 1), 0); del buf347  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (128, 2048), (2048, 1), 0), permute_593, out=buf354)
    del permute_593
    buf355 = reinterpret_tensor(buf352, (128, 2048), (2048, 1), 0); del buf352  # reuse
    cpp_fused_view_81(c_void_p(buf348.data_ptr()), c_void_p(buf355.data_ptr()))
    buf356 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf355, (2048, 128), (1, 2048), 0), view_310, out=buf356)
    del view_310
    buf357 = reinterpret_tensor(buf348, (128, 2048), (2048, 1), 0); del buf348  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf355, permute_597, out=buf357)
    del permute_597
    buf358 = buf336; del buf336  # reuse
    buf359 = buf335; del buf335  # reuse
    buf360 = reinterpret_tensor(buf345, (2048, ), (1, ), 0); del buf345  # reuse
    buf361 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf362 = buf339; del buf339  # reuse
    cpp_fused_add_native_layer_norm_backward_82(c_void_p(buf362.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(mul_112.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del div_46
    del mul_112
    del primals_185
    buf363 = reinterpret_tensor(buf331, (128, 8192), (8192, 1), 0); del buf331  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf362, (128, 2048), (2048, 1), 0), permute_599, out=buf363)
    del permute_599
    buf364 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf362, (2048, 128), (1, 2048), 0), view_308, out=buf364)
    del view_308
    buf365 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf366 = reinterpret_tensor(buf363, (1, 128, 8192), (1048576, 8192, 1), 0); del buf363  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_83(c_void_p(buf366.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(tanh_13.data_ptr()), c_void_p(buf365.data_ptr()))
    del addmm_40
    del tanh_13
    buf367 = buf357; del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf366, (128, 8192), (8192, 1), 0), permute_603, out=buf367)
    del permute_603
    buf368 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf366, (8192, 128), (1, 8192), 0), view_306, out=buf368)
    del view_306
    buf369 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf370 = buf359; del buf359  # reuse
    buf371 = buf358; del buf358  # reuse
    buf372 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf373 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf374 = buf362; del buf362  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_84(c_void_p(buf374.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(mul_106.data_ptr()), c_void_p(div_47.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()))
    del div_47
    del mul_106
    del primals_179
    buf375 = buf367; del buf367  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (128, 2048), (2048, 1), 0), permute_607, out=buf375)
    del permute_607
    buf376 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (2048, 128), (1, 2048), 0), view_304, out=buf376)
    del view_304
    buf377 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_85(c_void_p(buf374.data_ptr()), c_void_p(buf377.data_ptr()))
    buf378 = reinterpret_tensor(buf354, (16, 128, 128), (16384, 128, 1), 0); del buf354  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_612, reinterpret_tensor(buf375, (16, 128, 128), (128, 2048, 1), 0), out=buf378)
    del permute_612
    buf379 = reinterpret_tensor(buf351, (16, 128, 128), (16384, 128, 1), 0); del buf351  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf375, (16, 128, 128), (128, 2048, 1), 0), permute_613, out=buf379)
    del permute_613
    buf380 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf381 = reinterpret_tensor(buf379, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf379  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_86(c_void_p(buf381.data_ptr()), c_void_p(alias_71.data_ptr()), c_void_p(slice_56.data_ptr()), c_void_p(buf380.data_ptr()))
    del alias_71
    del slice_56
    buf382 = reinterpret_tensor(buf375, (16, 128, 128), (16384, 128, 1), 0); del buf375  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_614, reinterpret_tensor(buf381, (16, 128, 128), (16384, 128, 1), 0), out=buf382)
    del permute_614
    buf383 = reinterpret_tensor(buf355, (16, 128, 128), (16384, 128, 1), 0); del buf355  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf381, (16, 128, 128), (16384, 128, 1), 0), permute_615, out=buf383)
    del permute_615
    buf384 = reinterpret_tensor(buf381, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf381  # reuse
    cpp_fused_clone_87(c_void_p(tangents_30.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf384.data_ptr()))
    del tangents_30
    buf385 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf384, (2048, 128), (1, 2048), 0), view_288, out=buf385)
    buf386 = reinterpret_tensor(buf378, (128, 2048), (2048, 1), 0); del buf378  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf384, (128, 2048), (2048, 1), 0), permute_622, out=buf386)
    del permute_622
    buf387 = buf384; del buf384  # reuse
    cpp_fused_clone_88(c_void_p(tangents_29.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf387.data_ptr()))
    del tangents_29
    buf388 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf387, (2048, 128), (1, 2048), 0), view_288, out=buf388)
    buf389 = reinterpret_tensor(buf382, (128, 2048), (2048, 1), 0); del buf382  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf387, (128, 2048), (2048, 1), 0), permute_626, out=buf389)
    del permute_626
    buf390 = reinterpret_tensor(buf387, (128, 2048), (2048, 1), 0); del buf387  # reuse
    cpp_fused_view_89(c_void_p(buf383.data_ptr()), c_void_p(buf390.data_ptr()))
    buf391 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (2048, 128), (1, 2048), 0), view_288, out=buf391)
    del view_288
    buf392 = reinterpret_tensor(buf383, (128, 2048), (2048, 1), 0); del buf383  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf390, permute_630, out=buf392)
    del permute_630
    buf393 = buf371; del buf371  # reuse
    buf394 = buf370; del buf370  # reuse
    buf395 = reinterpret_tensor(buf380, (2048, ), (1, ), 0); del buf380  # reuse
    buf396 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf397 = buf374; del buf374  # reuse
    cpp_fused_add_native_layer_norm_backward_90(c_void_p(buf397.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(mul_104.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del div_48
    del mul_104
    del primals_172
    buf398 = reinterpret_tensor(buf366, (128, 8192), (8192, 1), 0); del buf366  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf397, (128, 2048), (2048, 1), 0), permute_632, out=buf398)
    del permute_632
    buf399 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf397, (2048, 128), (1, 2048), 0), view_286, out=buf399)
    del view_286
    buf400 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf401 = reinterpret_tensor(buf398, (1, 128, 8192), (1048576, 8192, 1), 0); del buf398  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_91(c_void_p(buf401.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(addmm_37.data_ptr()), c_void_p(tanh_12.data_ptr()), c_void_p(buf400.data_ptr()))
    del addmm_37
    del tanh_12
    buf402 = buf392; del buf392  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf401, (128, 8192), (8192, 1), 0), permute_636, out=buf402)
    del permute_636
    buf403 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf401, (8192, 128), (1, 8192), 0), view_284, out=buf403)
    del view_284
    buf404 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf405 = buf394; del buf394  # reuse
    buf406 = buf393; del buf393  # reuse
    buf407 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf408 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf409 = buf397; del buf397  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_92(c_void_p(buf409.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(mul_98.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()))
    del div_49
    del mul_98
    del primals_166
    buf410 = buf402; del buf402  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf409, (128, 2048), (2048, 1), 0), permute_640, out=buf410)
    del permute_640
    buf411 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf409, (2048, 128), (1, 2048), 0), view_282, out=buf411)
    del view_282
    buf412 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_93(c_void_p(buf409.data_ptr()), c_void_p(buf412.data_ptr()))
    buf413 = reinterpret_tensor(buf389, (16, 128, 128), (16384, 128, 1), 0); del buf389  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_645, reinterpret_tensor(buf410, (16, 128, 128), (128, 2048, 1), 0), out=buf413)
    del permute_645
    buf414 = reinterpret_tensor(buf386, (16, 128, 128), (16384, 128, 1), 0); del buf386  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf410, (16, 128, 128), (128, 2048, 1), 0), permute_646, out=buf414)
    del permute_646
    buf415 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf416 = reinterpret_tensor(buf414, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf414  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_94(c_void_p(buf416.data_ptr()), c_void_p(alias_73.data_ptr()), c_void_p(slice_52.data_ptr()), c_void_p(buf415.data_ptr()))
    del alias_73
    del slice_52
    buf417 = reinterpret_tensor(buf410, (16, 128, 128), (16384, 128, 1), 0); del buf410  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_647, reinterpret_tensor(buf416, (16, 128, 128), (16384, 128, 1), 0), out=buf417)
    del permute_647
    buf418 = reinterpret_tensor(buf390, (16, 128, 128), (16384, 128, 1), 0); del buf390  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf416, (16, 128, 128), (16384, 128, 1), 0), permute_648, out=buf418)
    del permute_648
    buf419 = reinterpret_tensor(buf416, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf416  # reuse
    cpp_fused_clone_95(c_void_p(tangents_28.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf419.data_ptr()))
    del tangents_28
    buf420 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (2048, 128), (1, 2048), 0), view_266, out=buf420)
    buf421 = reinterpret_tensor(buf413, (128, 2048), (2048, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (128, 2048), (2048, 1), 0), permute_655, out=buf421)
    del permute_655
    buf422 = buf419; del buf419  # reuse
    cpp_fused_clone_96(c_void_p(tangents_27.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf422.data_ptr()))
    del tangents_27
    buf423 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (2048, 128), (1, 2048), 0), view_266, out=buf423)
    buf424 = reinterpret_tensor(buf417, (128, 2048), (2048, 1), 0); del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (128, 2048), (2048, 1), 0), permute_659, out=buf424)
    del permute_659
    buf425 = reinterpret_tensor(buf422, (128, 2048), (2048, 1), 0); del buf422  # reuse
    cpp_fused_view_97(c_void_p(buf418.data_ptr()), c_void_p(buf425.data_ptr()))
    buf426 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf425, (2048, 128), (1, 2048), 0), view_266, out=buf426)
    del view_266
    buf427 = reinterpret_tensor(buf418, (128, 2048), (2048, 1), 0); del buf418  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf425, permute_663, out=buf427)
    del permute_663
    buf428 = buf406; del buf406  # reuse
    buf429 = buf405; del buf405  # reuse
    buf430 = reinterpret_tensor(buf415, (2048, ), (1, ), 0); del buf415  # reuse
    buf431 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf432 = buf409; del buf409  # reuse
    cpp_fused_add_native_layer_norm_backward_98(c_void_p(buf432.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_50.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()))
    del div_50
    del mul_96
    del primals_159
    buf433 = reinterpret_tensor(buf401, (128, 8192), (8192, 1), 0); del buf401  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (128, 2048), (2048, 1), 0), permute_665, out=buf433)
    del permute_665
    buf434 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (2048, 128), (1, 2048), 0), view_264, out=buf434)
    del view_264
    buf435 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf436 = reinterpret_tensor(buf433, (1, 128, 8192), (1048576, 8192, 1), 0); del buf433  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_99(c_void_p(buf436.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(tanh_11.data_ptr()), c_void_p(buf435.data_ptr()))
    del addmm_34
    del tanh_11
    buf437 = buf427; del buf427  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (128, 8192), (8192, 1), 0), permute_669, out=buf437)
    del permute_669
    buf438 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (8192, 128), (1, 8192), 0), view_262, out=buf438)
    del view_262
    buf439 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf440 = buf429; del buf429  # reuse
    buf441 = buf428; del buf428  # reuse
    buf442 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf443 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf444 = buf432; del buf432  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_100(c_void_p(buf444.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()))
    del div_51
    del mul_90
    del primals_153
    buf445 = buf437; del buf437  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf444, (128, 2048), (2048, 1), 0), permute_673, out=buf445)
    del permute_673
    buf446 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf444, (2048, 128), (1, 2048), 0), view_260, out=buf446)
    del view_260
    buf447 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_101(c_void_p(buf444.data_ptr()), c_void_p(buf447.data_ptr()))
    buf448 = reinterpret_tensor(buf424, (16, 128, 128), (16384, 128, 1), 0); del buf424  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_678, reinterpret_tensor(buf445, (16, 128, 128), (128, 2048, 1), 0), out=buf448)
    del permute_678
    buf449 = reinterpret_tensor(buf421, (16, 128, 128), (16384, 128, 1), 0); del buf421  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf445, (16, 128, 128), (128, 2048, 1), 0), permute_679, out=buf449)
    del permute_679
    buf450 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf451 = reinterpret_tensor(buf449, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf449  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_102(c_void_p(buf451.data_ptr()), c_void_p(alias_75.data_ptr()), c_void_p(slice_48.data_ptr()), c_void_p(buf450.data_ptr()))
    del alias_75
    del slice_48
    buf452 = reinterpret_tensor(buf445, (16, 128, 128), (16384, 128, 1), 0); del buf445  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_680, reinterpret_tensor(buf451, (16, 128, 128), (16384, 128, 1), 0), out=buf452)
    del permute_680
    buf453 = reinterpret_tensor(buf425, (16, 128, 128), (16384, 128, 1), 0); del buf425  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf451, (16, 128, 128), (16384, 128, 1), 0), permute_681, out=buf453)
    del permute_681
    buf454 = reinterpret_tensor(buf451, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf451  # reuse
    cpp_fused_clone_103(c_void_p(tangents_26.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf454.data_ptr()))
    del tangents_26
    buf455 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (2048, 128), (1, 2048), 0), view_244, out=buf455)
    buf456 = reinterpret_tensor(buf448, (128, 2048), (2048, 1), 0); del buf448  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (128, 2048), (2048, 1), 0), permute_688, out=buf456)
    del permute_688
    buf457 = buf454; del buf454  # reuse
    cpp_fused_clone_104(c_void_p(tangents_25.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf457.data_ptr()))
    del tangents_25
    buf458 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf457, (2048, 128), (1, 2048), 0), view_244, out=buf458)
    buf459 = reinterpret_tensor(buf452, (128, 2048), (2048, 1), 0); del buf452  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf457, (128, 2048), (2048, 1), 0), permute_692, out=buf459)
    del permute_692
    buf460 = reinterpret_tensor(buf457, (128, 2048), (2048, 1), 0); del buf457  # reuse
    cpp_fused_view_105(c_void_p(buf453.data_ptr()), c_void_p(buf460.data_ptr()))
    buf461 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf460, (2048, 128), (1, 2048), 0), view_244, out=buf461)
    del view_244
    buf462 = reinterpret_tensor(buf453, (128, 2048), (2048, 1), 0); del buf453  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf460, permute_696, out=buf462)
    del permute_696
    buf463 = buf441; del buf441  # reuse
    buf464 = buf440; del buf440  # reuse
    buf465 = reinterpret_tensor(buf450, (2048, ), (1, ), 0); del buf450  # reuse
    buf466 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf467 = buf444; del buf444  # reuse
    cpp_fused_add_native_layer_norm_backward_106(c_void_p(buf467.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    del div_52
    del mul_88
    del primals_146
    buf468 = reinterpret_tensor(buf436, (128, 8192), (8192, 1), 0); del buf436  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (128, 2048), (2048, 1), 0), permute_698, out=buf468)
    del permute_698
    buf469 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (2048, 128), (1, 2048), 0), view_242, out=buf469)
    del view_242
    buf470 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf471 = reinterpret_tensor(buf468, (1, 128, 8192), (1048576, 8192, 1), 0); del buf468  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_107(c_void_p(buf471.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(tanh_10.data_ptr()), c_void_p(buf470.data_ptr()))
    del addmm_31
    del tanh_10
    buf472 = buf462; del buf462  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf471, (128, 8192), (8192, 1), 0), permute_702, out=buf472)
    del permute_702
    buf473 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf471, (8192, 128), (1, 8192), 0), view_240, out=buf473)
    del view_240
    buf474 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf475 = buf464; del buf464  # reuse
    buf476 = buf463; del buf463  # reuse
    buf477 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf478 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf479 = buf467; del buf467  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_108(c_void_p(buf479.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(mul_82.data_ptr()), c_void_p(div_53.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()))
    del div_53
    del mul_82
    del primals_140
    buf480 = buf472; del buf472  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf479, (128, 2048), (2048, 1), 0), permute_706, out=buf480)
    del permute_706
    buf481 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf479, (2048, 128), (1, 2048), 0), view_238, out=buf481)
    del view_238
    buf482 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_109(c_void_p(buf479.data_ptr()), c_void_p(buf482.data_ptr()))
    buf483 = reinterpret_tensor(buf459, (16, 128, 128), (16384, 128, 1), 0); del buf459  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_711, reinterpret_tensor(buf480, (16, 128, 128), (128, 2048, 1), 0), out=buf483)
    del permute_711
    buf484 = reinterpret_tensor(buf456, (16, 128, 128), (16384, 128, 1), 0); del buf456  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf480, (16, 128, 128), (128, 2048, 1), 0), permute_712, out=buf484)
    del permute_712
    buf485 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf486 = reinterpret_tensor(buf484, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf484  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_110(c_void_p(buf486.data_ptr()), c_void_p(alias_77.data_ptr()), c_void_p(slice_44.data_ptr()), c_void_p(buf485.data_ptr()))
    del alias_77
    del slice_44
    buf487 = reinterpret_tensor(buf480, (16, 128, 128), (16384, 128, 1), 0); del buf480  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_713, reinterpret_tensor(buf486, (16, 128, 128), (16384, 128, 1), 0), out=buf487)
    del permute_713
    buf488 = reinterpret_tensor(buf460, (16, 128, 128), (16384, 128, 1), 0); del buf460  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf486, (16, 128, 128), (16384, 128, 1), 0), permute_714, out=buf488)
    del permute_714
    buf489 = reinterpret_tensor(buf486, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf486  # reuse
    cpp_fused_clone_111(c_void_p(tangents_24.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf489.data_ptr()))
    del tangents_24
    buf490 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf489, (2048, 128), (1, 2048), 0), view_222, out=buf490)
    buf491 = reinterpret_tensor(buf483, (128, 2048), (2048, 1), 0); del buf483  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf489, (128, 2048), (2048, 1), 0), permute_721, out=buf491)
    del permute_721
    buf492 = buf489; del buf489  # reuse
    cpp_fused_clone_112(c_void_p(tangents_23.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf492.data_ptr()))
    del tangents_23
    buf493 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf492, (2048, 128), (1, 2048), 0), view_222, out=buf493)
    buf494 = reinterpret_tensor(buf487, (128, 2048), (2048, 1), 0); del buf487  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf492, (128, 2048), (2048, 1), 0), permute_725, out=buf494)
    del permute_725
    buf495 = reinterpret_tensor(buf492, (128, 2048), (2048, 1), 0); del buf492  # reuse
    cpp_fused_view_113(c_void_p(buf488.data_ptr()), c_void_p(buf495.data_ptr()))
    buf496 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (2048, 128), (1, 2048), 0), view_222, out=buf496)
    del view_222
    buf497 = reinterpret_tensor(buf488, (128, 2048), (2048, 1), 0); del buf488  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf495, permute_729, out=buf497)
    del permute_729
    buf498 = buf476; del buf476  # reuse
    buf499 = buf475; del buf475  # reuse
    buf500 = reinterpret_tensor(buf485, (2048, ), (1, ), 0); del buf485  # reuse
    buf501 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf502 = buf479; del buf479  # reuse
    cpp_fused_add_native_layer_norm_backward_114(c_void_p(buf502.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    del div_54
    del mul_80
    del primals_133
    buf503 = reinterpret_tensor(buf471, (128, 8192), (8192, 1), 0); del buf471  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf502, (128, 2048), (2048, 1), 0), permute_731, out=buf503)
    del permute_731
    buf504 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf502, (2048, 128), (1, 2048), 0), view_220, out=buf504)
    del view_220
    buf505 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf506 = reinterpret_tensor(buf503, (1, 128, 8192), (1048576, 8192, 1), 0); del buf503  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_115(c_void_p(buf506.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(tanh_9.data_ptr()), c_void_p(buf505.data_ptr()))
    del addmm_28
    del tanh_9
    buf507 = buf497; del buf497  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf506, (128, 8192), (8192, 1), 0), permute_735, out=buf507)
    del permute_735
    buf508 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf506, (8192, 128), (1, 8192), 0), view_218, out=buf508)
    del view_218
    buf509 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf510 = buf499; del buf499  # reuse
    buf511 = buf498; del buf498  # reuse
    buf512 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf513 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf514 = buf502; del buf502  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_116(c_void_p(buf514.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(mul_74.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()))
    del div_55
    del mul_74
    del primals_127
    buf515 = buf507; del buf507  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf514, (128, 2048), (2048, 1), 0), permute_739, out=buf515)
    del permute_739
    buf516 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf514, (2048, 128), (1, 2048), 0), view_216, out=buf516)
    del view_216
    buf517 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_117(c_void_p(buf514.data_ptr()), c_void_p(buf517.data_ptr()))
    buf518 = reinterpret_tensor(buf494, (16, 128, 128), (16384, 128, 1), 0); del buf494  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_744, reinterpret_tensor(buf515, (16, 128, 128), (128, 2048, 1), 0), out=buf518)
    del permute_744
    buf519 = reinterpret_tensor(buf491, (16, 128, 128), (16384, 128, 1), 0); del buf491  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf515, (16, 128, 128), (128, 2048, 1), 0), permute_745, out=buf519)
    del permute_745
    buf520 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf521 = reinterpret_tensor(buf519, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf519  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_118(c_void_p(buf521.data_ptr()), c_void_p(alias_79.data_ptr()), c_void_p(slice_40.data_ptr()), c_void_p(buf520.data_ptr()))
    del alias_79
    del slice_40
    buf522 = reinterpret_tensor(buf515, (16, 128, 128), (16384, 128, 1), 0); del buf515  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_746, reinterpret_tensor(buf521, (16, 128, 128), (16384, 128, 1), 0), out=buf522)
    del permute_746
    buf523 = reinterpret_tensor(buf495, (16, 128, 128), (16384, 128, 1), 0); del buf495  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf521, (16, 128, 128), (16384, 128, 1), 0), permute_747, out=buf523)
    del permute_747
    buf524 = reinterpret_tensor(buf521, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf521  # reuse
    cpp_fused_clone_119(c_void_p(tangents_22.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf524.data_ptr()))
    del tangents_22
    buf525 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (2048, 128), (1, 2048), 0), view_200, out=buf525)
    buf526 = reinterpret_tensor(buf518, (128, 2048), (2048, 1), 0); del buf518  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (128, 2048), (2048, 1), 0), permute_754, out=buf526)
    del permute_754
    buf527 = buf524; del buf524  # reuse
    cpp_fused_clone_120(c_void_p(tangents_21.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf527.data_ptr()))
    del tangents_21
    buf528 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf527, (2048, 128), (1, 2048), 0), view_200, out=buf528)
    buf529 = reinterpret_tensor(buf522, (128, 2048), (2048, 1), 0); del buf522  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf527, (128, 2048), (2048, 1), 0), permute_758, out=buf529)
    del permute_758
    buf530 = reinterpret_tensor(buf527, (128, 2048), (2048, 1), 0); del buf527  # reuse
    cpp_fused_view_121(c_void_p(buf523.data_ptr()), c_void_p(buf530.data_ptr()))
    buf531 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf530, (2048, 128), (1, 2048), 0), view_200, out=buf531)
    del view_200
    buf532 = reinterpret_tensor(buf523, (128, 2048), (2048, 1), 0); del buf523  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf530, permute_762, out=buf532)
    del permute_762
    buf533 = buf511; del buf511  # reuse
    buf534 = buf510; del buf510  # reuse
    buf535 = reinterpret_tensor(buf520, (2048, ), (1, ), 0); del buf520  # reuse
    buf536 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf537 = buf514; del buf514  # reuse
    cpp_fused_add_native_layer_norm_backward_122(c_void_p(buf537.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_56.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()))
    del div_56
    del mul_72
    del primals_120
    buf538 = reinterpret_tensor(buf506, (128, 8192), (8192, 1), 0); del buf506  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (128, 2048), (2048, 1), 0), permute_764, out=buf538)
    del permute_764
    buf539 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (2048, 128), (1, 2048), 0), view_198, out=buf539)
    del view_198
    buf540 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf541 = reinterpret_tensor(buf538, (1, 128, 8192), (1048576, 8192, 1), 0); del buf538  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_123(c_void_p(buf541.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(tanh_8.data_ptr()), c_void_p(buf540.data_ptr()))
    del addmm_25
    del tanh_8
    buf542 = buf532; del buf532  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf541, (128, 8192), (8192, 1), 0), permute_768, out=buf542)
    del permute_768
    buf543 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf541, (8192, 128), (1, 8192), 0), view_196, out=buf543)
    del view_196
    buf544 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf545 = buf534; del buf534  # reuse
    buf546 = buf533; del buf533  # reuse
    buf547 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf548 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf549 = buf537; del buf537  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_124(c_void_p(buf549.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()))
    del div_57
    del mul_66
    del primals_114
    buf550 = buf542; del buf542  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf549, (128, 2048), (2048, 1), 0), permute_772, out=buf550)
    del permute_772
    buf551 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf549, (2048, 128), (1, 2048), 0), view_194, out=buf551)
    del view_194
    buf552 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_125(c_void_p(buf549.data_ptr()), c_void_p(buf552.data_ptr()))
    buf553 = reinterpret_tensor(buf529, (16, 128, 128), (16384, 128, 1), 0); del buf529  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_777, reinterpret_tensor(buf550, (16, 128, 128), (128, 2048, 1), 0), out=buf553)
    del permute_777
    buf554 = reinterpret_tensor(buf526, (16, 128, 128), (16384, 128, 1), 0); del buf526  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf550, (16, 128, 128), (128, 2048, 1), 0), permute_778, out=buf554)
    del permute_778
    buf555 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf556 = reinterpret_tensor(buf554, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf554  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_126(c_void_p(buf556.data_ptr()), c_void_p(alias_81.data_ptr()), c_void_p(slice_36.data_ptr()), c_void_p(buf555.data_ptr()))
    del alias_81
    del slice_36
    buf557 = reinterpret_tensor(buf550, (16, 128, 128), (16384, 128, 1), 0); del buf550  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_779, reinterpret_tensor(buf556, (16, 128, 128), (16384, 128, 1), 0), out=buf557)
    del permute_779
    buf558 = reinterpret_tensor(buf530, (16, 128, 128), (16384, 128, 1), 0); del buf530  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf556, (16, 128, 128), (16384, 128, 1), 0), permute_780, out=buf558)
    del permute_780
    buf559 = reinterpret_tensor(buf556, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf556  # reuse
    cpp_fused_clone_127(c_void_p(tangents_20.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf559.data_ptr()))
    del tangents_20
    buf560 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf559, (2048, 128), (1, 2048), 0), view_178, out=buf560)
    buf561 = reinterpret_tensor(buf553, (128, 2048), (2048, 1), 0); del buf553  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf559, (128, 2048), (2048, 1), 0), permute_787, out=buf561)
    del permute_787
    buf562 = buf559; del buf559  # reuse
    cpp_fused_clone_128(c_void_p(tangents_19.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf562.data_ptr()))
    del tangents_19
    buf563 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf562, (2048, 128), (1, 2048), 0), view_178, out=buf563)
    buf564 = reinterpret_tensor(buf557, (128, 2048), (2048, 1), 0); del buf557  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf562, (128, 2048), (2048, 1), 0), permute_791, out=buf564)
    del permute_791
    buf565 = reinterpret_tensor(buf562, (128, 2048), (2048, 1), 0); del buf562  # reuse
    cpp_fused_view_129(c_void_p(buf558.data_ptr()), c_void_p(buf565.data_ptr()))
    buf566 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf565, (2048, 128), (1, 2048), 0), view_178, out=buf566)
    del view_178
    buf567 = reinterpret_tensor(buf558, (128, 2048), (2048, 1), 0); del buf558  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf565, permute_795, out=buf567)
    del permute_795
    buf568 = buf546; del buf546  # reuse
    buf569 = buf545; del buf545  # reuse
    buf570 = reinterpret_tensor(buf555, (2048, ), (1, ), 0); del buf555  # reuse
    buf571 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf572 = buf549; del buf549  # reuse
    cpp_fused_add_native_layer_norm_backward_130(c_void_p(buf572.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()))
    del div_58
    del mul_64
    del primals_107
    buf573 = reinterpret_tensor(buf541, (128, 8192), (8192, 1), 0); del buf541  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf572, (128, 2048), (2048, 1), 0), permute_797, out=buf573)
    del permute_797
    buf574 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf572, (2048, 128), (1, 2048), 0), view_176, out=buf574)
    del view_176
    buf575 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf576 = reinterpret_tensor(buf573, (1, 128, 8192), (1048576, 8192, 1), 0); del buf573  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_131(c_void_p(buf576.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(tanh_7.data_ptr()), c_void_p(buf575.data_ptr()))
    del addmm_22
    del tanh_7
    buf577 = buf567; del buf567  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf576, (128, 8192), (8192, 1), 0), permute_801, out=buf577)
    del permute_801
    buf578 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf576, (8192, 128), (1, 8192), 0), view_174, out=buf578)
    del view_174
    buf579 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf580 = buf569; del buf569  # reuse
    buf581 = buf568; del buf568  # reuse
    buf582 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf583 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf584 = buf572; del buf572  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_132(c_void_p(buf584.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(mul_58.data_ptr()), c_void_p(div_59.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()))
    del div_59
    del mul_58
    del primals_101
    buf585 = buf577; del buf577  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf584, (128, 2048), (2048, 1), 0), permute_805, out=buf585)
    del permute_805
    buf586 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf584, (2048, 128), (1, 2048), 0), view_172, out=buf586)
    del view_172
    buf587 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_133(c_void_p(buf584.data_ptr()), c_void_p(buf587.data_ptr()))
    buf588 = reinterpret_tensor(buf564, (16, 128, 128), (16384, 128, 1), 0); del buf564  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_810, reinterpret_tensor(buf585, (16, 128, 128), (128, 2048, 1), 0), out=buf588)
    del permute_810
    buf589 = reinterpret_tensor(buf561, (16, 128, 128), (16384, 128, 1), 0); del buf561  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf585, (16, 128, 128), (128, 2048, 1), 0), permute_811, out=buf589)
    del permute_811
    buf590 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf591 = reinterpret_tensor(buf589, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf589  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_134(c_void_p(buf591.data_ptr()), c_void_p(alias_83.data_ptr()), c_void_p(slice_32.data_ptr()), c_void_p(buf590.data_ptr()))
    del alias_83
    del slice_32
    buf592 = reinterpret_tensor(buf585, (16, 128, 128), (16384, 128, 1), 0); del buf585  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_812, reinterpret_tensor(buf591, (16, 128, 128), (16384, 128, 1), 0), out=buf592)
    del permute_812
    buf593 = reinterpret_tensor(buf565, (16, 128, 128), (16384, 128, 1), 0); del buf565  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf591, (16, 128, 128), (16384, 128, 1), 0), permute_813, out=buf593)
    del permute_813
    buf594 = reinterpret_tensor(buf591, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf591  # reuse
    cpp_fused_clone_135(c_void_p(tangents_18.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf594.data_ptr()))
    del tangents_18
    buf595 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf594, (2048, 128), (1, 2048), 0), view_156, out=buf595)
    buf596 = reinterpret_tensor(buf588, (128, 2048), (2048, 1), 0); del buf588  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf594, (128, 2048), (2048, 1), 0), permute_820, out=buf596)
    del permute_820
    buf597 = buf594; del buf594  # reuse
    cpp_fused_clone_136(c_void_p(tangents_17.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf597.data_ptr()))
    del tangents_17
    buf598 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf597, (2048, 128), (1, 2048), 0), view_156, out=buf598)
    buf599 = reinterpret_tensor(buf592, (128, 2048), (2048, 1), 0); del buf592  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf597, (128, 2048), (2048, 1), 0), permute_824, out=buf599)
    del permute_824
    buf600 = reinterpret_tensor(buf597, (128, 2048), (2048, 1), 0); del buf597  # reuse
    cpp_fused_view_137(c_void_p(buf593.data_ptr()), c_void_p(buf600.data_ptr()))
    buf601 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf600, (2048, 128), (1, 2048), 0), view_156, out=buf601)
    del view_156
    buf602 = reinterpret_tensor(buf593, (128, 2048), (2048, 1), 0); del buf593  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf600, permute_828, out=buf602)
    del permute_828
    buf603 = buf581; del buf581  # reuse
    buf604 = buf580; del buf580  # reuse
    buf605 = reinterpret_tensor(buf590, (2048, ), (1, ), 0); del buf590  # reuse
    buf606 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf607 = buf584; del buf584  # reuse
    cpp_fused_add_native_layer_norm_backward_138(c_void_p(buf607.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    del div_60
    del mul_56
    del primals_94
    buf608 = reinterpret_tensor(buf576, (128, 8192), (8192, 1), 0); del buf576  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf607, (128, 2048), (2048, 1), 0), permute_830, out=buf608)
    del permute_830
    buf609 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf607, (2048, 128), (1, 2048), 0), view_154, out=buf609)
    del view_154
    buf610 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf611 = reinterpret_tensor(buf608, (1, 128, 8192), (1048576, 8192, 1), 0); del buf608  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_139(c_void_p(buf611.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(tanh_6.data_ptr()), c_void_p(buf610.data_ptr()))
    del addmm_19
    del tanh_6
    buf612 = buf602; del buf602  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf611, (128, 8192), (8192, 1), 0), permute_834, out=buf612)
    del permute_834
    buf613 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf611, (8192, 128), (1, 8192), 0), view_152, out=buf613)
    del view_152
    buf614 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf615 = buf604; del buf604  # reuse
    buf616 = buf603; del buf603  # reuse
    buf617 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf618 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf619 = buf607; del buf607  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_140(c_void_p(buf619.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf618.data_ptr()))
    del div_61
    del mul_50
    del primals_88
    buf620 = buf612; del buf612  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf619, (128, 2048), (2048, 1), 0), permute_838, out=buf620)
    del permute_838
    buf621 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf619, (2048, 128), (1, 2048), 0), view_150, out=buf621)
    del view_150
    buf622 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_141(c_void_p(buf619.data_ptr()), c_void_p(buf622.data_ptr()))
    buf623 = reinterpret_tensor(buf599, (16, 128, 128), (16384, 128, 1), 0); del buf599  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_843, reinterpret_tensor(buf620, (16, 128, 128), (128, 2048, 1), 0), out=buf623)
    del permute_843
    buf624 = reinterpret_tensor(buf596, (16, 128, 128), (16384, 128, 1), 0); del buf596  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf620, (16, 128, 128), (128, 2048, 1), 0), permute_844, out=buf624)
    del permute_844
    buf625 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf626 = reinterpret_tensor(buf624, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf624  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_142(c_void_p(buf626.data_ptr()), c_void_p(alias_85.data_ptr()), c_void_p(slice_28.data_ptr()), c_void_p(buf625.data_ptr()))
    del alias_85
    del slice_28
    buf627 = reinterpret_tensor(buf620, (16, 128, 128), (16384, 128, 1), 0); del buf620  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_845, reinterpret_tensor(buf626, (16, 128, 128), (16384, 128, 1), 0), out=buf627)
    del permute_845
    buf628 = reinterpret_tensor(buf600, (16, 128, 128), (16384, 128, 1), 0); del buf600  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf626, (16, 128, 128), (16384, 128, 1), 0), permute_846, out=buf628)
    del permute_846
    buf629 = reinterpret_tensor(buf626, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf626  # reuse
    cpp_fused_clone_143(c_void_p(tangents_16.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf629.data_ptr()))
    del tangents_16
    buf630 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (2048, 128), (1, 2048), 0), view_134, out=buf630)
    buf631 = reinterpret_tensor(buf623, (128, 2048), (2048, 1), 0); del buf623  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (128, 2048), (2048, 1), 0), permute_853, out=buf631)
    del permute_853
    buf632 = buf629; del buf629  # reuse
    cpp_fused_clone_144(c_void_p(tangents_15.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf632.data_ptr()))
    del tangents_15
    buf633 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf632, (2048, 128), (1, 2048), 0), view_134, out=buf633)
    buf634 = reinterpret_tensor(buf627, (128, 2048), (2048, 1), 0); del buf627  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf632, (128, 2048), (2048, 1), 0), permute_857, out=buf634)
    del permute_857
    buf635 = reinterpret_tensor(buf632, (128, 2048), (2048, 1), 0); del buf632  # reuse
    cpp_fused_view_145(c_void_p(buf628.data_ptr()), c_void_p(buf635.data_ptr()))
    buf636 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf635, (2048, 128), (1, 2048), 0), view_134, out=buf636)
    del view_134
    buf637 = reinterpret_tensor(buf628, (128, 2048), (2048, 1), 0); del buf628  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf635, permute_861, out=buf637)
    del permute_861
    buf638 = buf616; del buf616  # reuse
    buf639 = buf615; del buf615  # reuse
    buf640 = reinterpret_tensor(buf625, (2048, ), (1, ), 0); del buf625  # reuse
    buf641 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf642 = buf619; del buf619  # reuse
    cpp_fused_add_native_layer_norm_backward_146(c_void_p(buf642.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_62.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()))
    del div_62
    del mul_48
    del primals_81
    buf643 = reinterpret_tensor(buf611, (128, 8192), (8192, 1), 0); del buf611  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf642, (128, 2048), (2048, 1), 0), permute_863, out=buf643)
    del permute_863
    buf644 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf642, (2048, 128), (1, 2048), 0), view_132, out=buf644)
    del view_132
    buf645 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf646 = reinterpret_tensor(buf643, (1, 128, 8192), (1048576, 8192, 1), 0); del buf643  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_147(c_void_p(buf646.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(tanh_5.data_ptr()), c_void_p(buf645.data_ptr()))
    del addmm_16
    del tanh_5
    buf647 = buf637; del buf637  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf646, (128, 8192), (8192, 1), 0), permute_867, out=buf647)
    del permute_867
    buf648 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf646, (8192, 128), (1, 8192), 0), view_130, out=buf648)
    del view_130
    buf649 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf650 = buf639; del buf639  # reuse
    buf651 = buf638; del buf638  # reuse
    buf652 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf653 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf654 = buf642; del buf642  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_148(c_void_p(buf654.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_63.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()))
    del div_63
    del mul_42
    del primals_75
    buf655 = buf647; del buf647  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf654, (128, 2048), (2048, 1), 0), permute_871, out=buf655)
    del permute_871
    buf656 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf654, (2048, 128), (1, 2048), 0), view_128, out=buf656)
    del view_128
    buf657 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_149(c_void_p(buf654.data_ptr()), c_void_p(buf657.data_ptr()))
    buf658 = reinterpret_tensor(buf634, (16, 128, 128), (16384, 128, 1), 0); del buf634  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_876, reinterpret_tensor(buf655, (16, 128, 128), (128, 2048, 1), 0), out=buf658)
    del permute_876
    buf659 = reinterpret_tensor(buf631, (16, 128, 128), (16384, 128, 1), 0); del buf631  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf655, (16, 128, 128), (128, 2048, 1), 0), permute_877, out=buf659)
    del permute_877
    buf660 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf661 = reinterpret_tensor(buf659, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf659  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_150(c_void_p(buf661.data_ptr()), c_void_p(alias_87.data_ptr()), c_void_p(slice_24.data_ptr()), c_void_p(buf660.data_ptr()))
    del alias_87
    del slice_24
    buf662 = reinterpret_tensor(buf655, (16, 128, 128), (16384, 128, 1), 0); del buf655  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_878, reinterpret_tensor(buf661, (16, 128, 128), (16384, 128, 1), 0), out=buf662)
    del permute_878
    buf663 = reinterpret_tensor(buf635, (16, 128, 128), (16384, 128, 1), 0); del buf635  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf661, (16, 128, 128), (16384, 128, 1), 0), permute_879, out=buf663)
    del permute_879
    buf664 = reinterpret_tensor(buf661, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf661  # reuse
    cpp_fused_clone_151(c_void_p(tangents_14.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf664.data_ptr()))
    del tangents_14
    buf665 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf664, (2048, 128), (1, 2048), 0), view_112, out=buf665)
    buf666 = reinterpret_tensor(buf658, (128, 2048), (2048, 1), 0); del buf658  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf664, (128, 2048), (2048, 1), 0), permute_886, out=buf666)
    del permute_886
    buf667 = buf664; del buf664  # reuse
    cpp_fused_clone_152(c_void_p(tangents_13.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf667.data_ptr()))
    del tangents_13
    buf668 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf667, (2048, 128), (1, 2048), 0), view_112, out=buf668)
    buf669 = reinterpret_tensor(buf662, (128, 2048), (2048, 1), 0); del buf662  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf667, (128, 2048), (2048, 1), 0), permute_890, out=buf669)
    del permute_890
    buf670 = reinterpret_tensor(buf667, (128, 2048), (2048, 1), 0); del buf667  # reuse
    cpp_fused_view_153(c_void_p(buf663.data_ptr()), c_void_p(buf670.data_ptr()))
    buf671 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf670, (2048, 128), (1, 2048), 0), view_112, out=buf671)
    del view_112
    buf672 = reinterpret_tensor(buf663, (128, 2048), (2048, 1), 0); del buf663  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf670, permute_894, out=buf672)
    del permute_894
    buf673 = buf651; del buf651  # reuse
    buf674 = buf650; del buf650  # reuse
    buf675 = reinterpret_tensor(buf660, (2048, ), (1, ), 0); del buf660  # reuse
    buf676 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf677 = buf654; del buf654  # reuse
    cpp_fused_add_native_layer_norm_backward_154(c_void_p(buf677.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_64.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()))
    del div_64
    del mul_40
    del primals_68
    buf678 = reinterpret_tensor(buf646, (128, 8192), (8192, 1), 0); del buf646  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf677, (128, 2048), (2048, 1), 0), permute_896, out=buf678)
    del permute_896
    buf679 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf677, (2048, 128), (1, 2048), 0), view_110, out=buf679)
    del view_110
    buf680 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf681 = reinterpret_tensor(buf678, (1, 128, 8192), (1048576, 8192, 1), 0); del buf678  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_155(c_void_p(buf681.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(tanh_4.data_ptr()), c_void_p(buf680.data_ptr()))
    del addmm_13
    del tanh_4
    buf682 = buf672; del buf672  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf681, (128, 8192), (8192, 1), 0), permute_900, out=buf682)
    del permute_900
    buf683 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf681, (8192, 128), (1, 8192), 0), view_108, out=buf683)
    del view_108
    buf684 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf685 = buf674; del buf674  # reuse
    buf686 = buf673; del buf673  # reuse
    buf687 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf688 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf689 = buf677; del buf677  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_156(c_void_p(buf689.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(mul_34.data_ptr()), c_void_p(div_65.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf688.data_ptr()))
    del div_65
    del mul_34
    del primals_62
    buf690 = buf682; del buf682  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf689, (128, 2048), (2048, 1), 0), permute_904, out=buf690)
    del permute_904
    buf691 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf689, (2048, 128), (1, 2048), 0), view_106, out=buf691)
    del view_106
    buf692 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_157(c_void_p(buf689.data_ptr()), c_void_p(buf692.data_ptr()))
    buf693 = reinterpret_tensor(buf669, (16, 128, 128), (16384, 128, 1), 0); del buf669  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_909, reinterpret_tensor(buf690, (16, 128, 128), (128, 2048, 1), 0), out=buf693)
    del permute_909
    buf694 = reinterpret_tensor(buf666, (16, 128, 128), (16384, 128, 1), 0); del buf666  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf690, (16, 128, 128), (128, 2048, 1), 0), permute_910, out=buf694)
    del permute_910
    buf695 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf696 = reinterpret_tensor(buf694, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf694  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_158(c_void_p(buf696.data_ptr()), c_void_p(alias_89.data_ptr()), c_void_p(slice_20.data_ptr()), c_void_p(buf695.data_ptr()))
    del alias_89
    del slice_20
    buf697 = reinterpret_tensor(buf690, (16, 128, 128), (16384, 128, 1), 0); del buf690  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_911, reinterpret_tensor(buf696, (16, 128, 128), (16384, 128, 1), 0), out=buf697)
    del permute_911
    buf698 = reinterpret_tensor(buf670, (16, 128, 128), (16384, 128, 1), 0); del buf670  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf696, (16, 128, 128), (16384, 128, 1), 0), permute_912, out=buf698)
    del permute_912
    buf699 = reinterpret_tensor(buf696, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf696  # reuse
    cpp_fused_clone_159(c_void_p(tangents_12.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf699.data_ptr()))
    del tangents_12
    buf700 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf699, (2048, 128), (1, 2048), 0), view_90, out=buf700)
    buf701 = reinterpret_tensor(buf693, (128, 2048), (2048, 1), 0); del buf693  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf699, (128, 2048), (2048, 1), 0), permute_919, out=buf701)
    del permute_919
    buf702 = buf699; del buf699  # reuse
    cpp_fused_clone_160(c_void_p(tangents_11.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf702.data_ptr()))
    del tangents_11
    buf703 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf702, (2048, 128), (1, 2048), 0), view_90, out=buf703)
    buf704 = reinterpret_tensor(buf697, (128, 2048), (2048, 1), 0); del buf697  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf702, (128, 2048), (2048, 1), 0), permute_923, out=buf704)
    del permute_923
    buf705 = reinterpret_tensor(buf702, (128, 2048), (2048, 1), 0); del buf702  # reuse
    cpp_fused_view_161(c_void_p(buf698.data_ptr()), c_void_p(buf705.data_ptr()))
    buf706 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf705, (2048, 128), (1, 2048), 0), view_90, out=buf706)
    del view_90
    buf707 = reinterpret_tensor(buf698, (128, 2048), (2048, 1), 0); del buf698  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf705, permute_927, out=buf707)
    del permute_927
    buf708 = buf686; del buf686  # reuse
    buf709 = buf685; del buf685  # reuse
    buf710 = reinterpret_tensor(buf695, (2048, ), (1, ), 0); del buf695  # reuse
    buf711 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf712 = buf689; del buf689  # reuse
    cpp_fused_add_native_layer_norm_backward_162(c_void_p(buf712.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()))
    del div_66
    del mul_32
    del primals_55
    buf713 = reinterpret_tensor(buf681, (128, 8192), (8192, 1), 0); del buf681  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf712, (128, 2048), (2048, 1), 0), permute_929, out=buf713)
    del permute_929
    buf714 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf712, (2048, 128), (1, 2048), 0), view_88, out=buf714)
    del view_88
    buf715 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf716 = reinterpret_tensor(buf713, (1, 128, 8192), (1048576, 8192, 1), 0); del buf713  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_163(c_void_p(buf716.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(tanh_3.data_ptr()), c_void_p(buf715.data_ptr()))
    del addmm_10
    del tanh_3
    buf717 = buf707; del buf707  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf716, (128, 8192), (8192, 1), 0), permute_933, out=buf717)
    del permute_933
    buf718 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf716, (8192, 128), (1, 8192), 0), view_86, out=buf718)
    del view_86
    buf719 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf720 = buf709; del buf709  # reuse
    buf721 = buf708; del buf708  # reuse
    buf722 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf723 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf724 = buf712; del buf712  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_164(c_void_p(buf724.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(mul_26.data_ptr()), c_void_p(div_67.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf723.data_ptr()))
    del div_67
    del mul_26
    del primals_49
    buf725 = buf717; del buf717  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf724, (128, 2048), (2048, 1), 0), permute_937, out=buf725)
    del permute_937
    buf726 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf724, (2048, 128), (1, 2048), 0), view_84, out=buf726)
    del view_84
    buf727 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_165(c_void_p(buf724.data_ptr()), c_void_p(buf727.data_ptr()))
    buf728 = reinterpret_tensor(buf704, (16, 128, 128), (16384, 128, 1), 0); del buf704  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_942, reinterpret_tensor(buf725, (16, 128, 128), (128, 2048, 1), 0), out=buf728)
    del permute_942
    buf729 = reinterpret_tensor(buf701, (16, 128, 128), (16384, 128, 1), 0); del buf701  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf725, (16, 128, 128), (128, 2048, 1), 0), permute_943, out=buf729)
    del permute_943
    buf730 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf731 = reinterpret_tensor(buf729, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf729  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_166(c_void_p(buf731.data_ptr()), c_void_p(alias_91.data_ptr()), c_void_p(slice_16.data_ptr()), c_void_p(buf730.data_ptr()))
    del alias_91
    del slice_16
    buf732 = reinterpret_tensor(buf725, (16, 128, 128), (16384, 128, 1), 0); del buf725  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_944, reinterpret_tensor(buf731, (16, 128, 128), (16384, 128, 1), 0), out=buf732)
    del permute_944
    buf733 = reinterpret_tensor(buf705, (16, 128, 128), (16384, 128, 1), 0); del buf705  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf731, (16, 128, 128), (16384, 128, 1), 0), permute_945, out=buf733)
    del permute_945
    buf734 = reinterpret_tensor(buf731, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf731  # reuse
    cpp_fused_clone_167(c_void_p(tangents_10.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf734.data_ptr()))
    del tangents_10
    buf735 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf734, (2048, 128), (1, 2048), 0), view_68, out=buf735)
    buf736 = reinterpret_tensor(buf728, (128, 2048), (2048, 1), 0); del buf728  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf734, (128, 2048), (2048, 1), 0), permute_952, out=buf736)
    del permute_952
    buf737 = buf734; del buf734  # reuse
    cpp_fused_clone_168(c_void_p(tangents_9.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf737.data_ptr()))
    del tangents_9
    buf738 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf737, (2048, 128), (1, 2048), 0), view_68, out=buf738)
    buf739 = reinterpret_tensor(buf732, (128, 2048), (2048, 1), 0); del buf732  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf737, (128, 2048), (2048, 1), 0), permute_956, out=buf739)
    del permute_956
    buf740 = reinterpret_tensor(buf737, (128, 2048), (2048, 1), 0); del buf737  # reuse
    cpp_fused_view_169(c_void_p(buf733.data_ptr()), c_void_p(buf740.data_ptr()))
    buf741 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf740, (2048, 128), (1, 2048), 0), view_68, out=buf741)
    del view_68
    buf742 = reinterpret_tensor(buf733, (128, 2048), (2048, 1), 0); del buf733  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf740, permute_960, out=buf742)
    del permute_960
    buf743 = buf721; del buf721  # reuse
    buf744 = buf720; del buf720  # reuse
    buf745 = reinterpret_tensor(buf730, (2048, ), (1, ), 0); del buf730  # reuse
    buf746 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf747 = buf724; del buf724  # reuse
    cpp_fused_add_native_layer_norm_backward_170(c_void_p(buf747.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_68.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()))
    del div_68
    del mul_24
    del primals_42
    buf748 = reinterpret_tensor(buf716, (128, 8192), (8192, 1), 0); del buf716  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf747, (128, 2048), (2048, 1), 0), permute_962, out=buf748)
    del permute_962
    buf749 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf747, (2048, 128), (1, 2048), 0), view_66, out=buf749)
    del view_66
    buf750 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf751 = reinterpret_tensor(buf748, (1, 128, 8192), (1048576, 8192, 1), 0); del buf748  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_171(c_void_p(buf751.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(tanh_2.data_ptr()), c_void_p(buf750.data_ptr()))
    del addmm_7
    del tanh_2
    buf752 = buf742; del buf742  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf751, (128, 8192), (8192, 1), 0), permute_966, out=buf752)
    del permute_966
    buf753 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf751, (8192, 128), (1, 8192), 0), view_64, out=buf753)
    del view_64
    buf754 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf755 = buf744; del buf744  # reuse
    buf756 = buf743; del buf743  # reuse
    buf757 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf758 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf759 = buf747; del buf747  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_172(c_void_p(buf759.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(mul_18.data_ptr()), c_void_p(div_69.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf758.data_ptr()))
    del div_69
    del mul_18
    del primals_36
    buf760 = buf752; del buf752  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf759, (128, 2048), (2048, 1), 0), permute_970, out=buf760)
    del permute_970
    buf761 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf759, (2048, 128), (1, 2048), 0), view_62, out=buf761)
    del view_62
    buf762 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_173(c_void_p(buf759.data_ptr()), c_void_p(buf762.data_ptr()))
    buf763 = reinterpret_tensor(buf739, (16, 128, 128), (16384, 128, 1), 0); del buf739  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_975, reinterpret_tensor(buf760, (16, 128, 128), (128, 2048, 1), 0), out=buf763)
    del permute_975
    buf764 = reinterpret_tensor(buf736, (16, 128, 128), (16384, 128, 1), 0); del buf736  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf760, (16, 128, 128), (128, 2048, 1), 0), permute_976, out=buf764)
    del permute_976
    buf765 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf766 = reinterpret_tensor(buf764, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf764  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_174(c_void_p(buf766.data_ptr()), c_void_p(alias_93.data_ptr()), c_void_p(slice_12.data_ptr()), c_void_p(buf765.data_ptr()))
    del alias_93
    del slice_12
    buf767 = reinterpret_tensor(buf760, (16, 128, 128), (16384, 128, 1), 0); del buf760  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_977, reinterpret_tensor(buf766, (16, 128, 128), (16384, 128, 1), 0), out=buf767)
    del permute_977
    buf768 = reinterpret_tensor(buf740, (16, 128, 128), (16384, 128, 1), 0); del buf740  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf766, (16, 128, 128), (16384, 128, 1), 0), permute_978, out=buf768)
    del permute_978
    buf769 = reinterpret_tensor(buf766, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf766  # reuse
    cpp_fused_clone_175(c_void_p(tangents_8.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf769.data_ptr()))
    del tangents_8
    buf770 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf769, (2048, 128), (1, 2048), 0), view_46, out=buf770)
    buf771 = reinterpret_tensor(buf763, (128, 2048), (2048, 1), 0); del buf763  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf769, (128, 2048), (2048, 1), 0), permute_985, out=buf771)
    del permute_985
    buf772 = buf769; del buf769  # reuse
    cpp_fused_clone_176(c_void_p(tangents_7.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf772.data_ptr()))
    del tangents_7
    buf773 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf772, (2048, 128), (1, 2048), 0), view_46, out=buf773)
    buf774 = reinterpret_tensor(buf767, (128, 2048), (2048, 1), 0); del buf767  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf772, (128, 2048), (2048, 1), 0), permute_989, out=buf774)
    del permute_989
    buf775 = reinterpret_tensor(buf772, (128, 2048), (2048, 1), 0); del buf772  # reuse
    cpp_fused_view_177(c_void_p(buf768.data_ptr()), c_void_p(buf775.data_ptr()))
    buf776 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf775, (2048, 128), (1, 2048), 0), view_46, out=buf776)
    del view_46
    buf777 = reinterpret_tensor(buf768, (128, 2048), (2048, 1), 0); del buf768  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf775, permute_993, out=buf777)
    del permute_993
    buf778 = buf756; del buf756  # reuse
    buf779 = buf755; del buf755  # reuse
    buf780 = reinterpret_tensor(buf765, (2048, ), (1, ), 0); del buf765  # reuse
    buf781 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf782 = buf759; del buf759  # reuse
    cpp_fused_add_native_layer_norm_backward_178(c_void_p(buf782.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_70.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf779.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(buf781.data_ptr()))
    del div_70
    del mul_16
    del primals_29
    buf783 = reinterpret_tensor(buf751, (128, 8192), (8192, 1), 0); del buf751  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf782, (128, 2048), (2048, 1), 0), permute_995, out=buf783)
    del permute_995
    buf784 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf782, (2048, 128), (1, 2048), 0), view_44, out=buf784)
    del view_44
    buf785 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf786 = reinterpret_tensor(buf783, (1, 128, 8192), (1048576, 8192, 1), 0); del buf783  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_179(c_void_p(buf786.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(tanh_1.data_ptr()), c_void_p(buf785.data_ptr()))
    del addmm_4
    del tanh_1
    buf787 = buf777; del buf777  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf786, (128, 8192), (8192, 1), 0), permute_999, out=buf787)
    del permute_999
    buf788 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf786, (8192, 128), (1, 8192), 0), view_42, out=buf788)
    del view_42
    buf789 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf790 = buf779; del buf779  # reuse
    buf791 = buf778; del buf778  # reuse
    buf792 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf793 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf794 = buf782; del buf782  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_180(c_void_p(buf794.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_71.data_ptr()), c_void_p(buf789.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf793.data_ptr()))
    del div_71
    del mul_10
    del primals_23
    buf795 = buf787; del buf787  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf794, (128, 2048), (2048, 1), 0), permute_1003, out=buf795)
    del permute_1003
    buf796 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf794, (2048, 128), (1, 2048), 0), view_40, out=buf796)
    del view_40
    buf797 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_181(c_void_p(buf794.data_ptr()), c_void_p(buf797.data_ptr()))
    buf798 = reinterpret_tensor(buf774, (16, 128, 128), (16384, 128, 1), 0); del buf774  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1008, reinterpret_tensor(buf795, (16, 128, 128), (128, 2048, 1), 0), out=buf798)
    del permute_1008
    buf799 = reinterpret_tensor(buf771, (16, 128, 128), (16384, 128, 1), 0); del buf771  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf795, (16, 128, 128), (128, 2048, 1), 0), permute_1009, out=buf799)
    del permute_1009
    buf800 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf801 = reinterpret_tensor(buf799, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf799  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_182(c_void_p(buf801.data_ptr()), c_void_p(alias_95.data_ptr()), c_void_p(slice_8.data_ptr()), c_void_p(buf800.data_ptr()))
    del alias_95
    del slice_8
    buf802 = reinterpret_tensor(buf795, (16, 128, 128), (16384, 128, 1), 0); del buf795  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1010, reinterpret_tensor(buf801, (16, 128, 128), (16384, 128, 1), 0), out=buf802)
    del permute_1010
    buf803 = reinterpret_tensor(buf775, (16, 128, 128), (16384, 128, 1), 0); del buf775  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf801, (16, 128, 128), (16384, 128, 1), 0), permute_1011, out=buf803)
    del permute_1011
    buf804 = reinterpret_tensor(buf801, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf801  # reuse
    cpp_fused_clone_183(c_void_p(tangents_6.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf804.data_ptr()))
    del tangents_6
    buf805 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf804, (2048, 128), (1, 2048), 0), view_24, out=buf805)
    buf806 = reinterpret_tensor(buf798, (128, 2048), (2048, 1), 0); del buf798  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf804, (128, 2048), (2048, 1), 0), permute_1018, out=buf806)
    del permute_1018
    buf807 = buf804; del buf804  # reuse
    cpp_fused_clone_184(c_void_p(tangents_5.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf807.data_ptr()))
    del tangents_5
    buf808 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf807, (2048, 128), (1, 2048), 0), view_24, out=buf808)
    buf809 = reinterpret_tensor(buf802, (128, 2048), (2048, 1), 0); del buf802  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf807, (128, 2048), (2048, 1), 0), permute_1022, out=buf809)
    del permute_1022
    buf810 = reinterpret_tensor(buf807, (128, 2048), (2048, 1), 0); del buf807  # reuse
    cpp_fused_view_185(c_void_p(buf803.data_ptr()), c_void_p(buf810.data_ptr()))
    buf811 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf810, (2048, 128), (1, 2048), 0), view_24, out=buf811)
    del view_24
    buf812 = reinterpret_tensor(buf803, (128, 2048), (2048, 1), 0); del buf803  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf810, permute_1026, out=buf812)
    del permute_1026
    buf813 = buf791; del buf791  # reuse
    buf814 = buf790; del buf790  # reuse
    buf815 = reinterpret_tensor(buf800, (2048, ), (1, ), 0); del buf800  # reuse
    buf816 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf817 = buf794; del buf794  # reuse
    cpp_fused_add_native_layer_norm_backward_186(c_void_p(buf817.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_72.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf816.data_ptr()))
    del div_72
    del mul_8
    del primals_16
    buf818 = reinterpret_tensor(buf786, (128, 8192), (8192, 1), 0); del buf786  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf817, (128, 2048), (2048, 1), 0), permute_1028, out=buf818)
    del permute_1028
    buf819 = empty((2048, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf817, (2048, 128), (1, 2048), 0), view_22, out=buf819)
    del view_22
    buf820 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf821 = reinterpret_tensor(buf818, (1, 128, 8192), (1048576, 8192, 1), 0); del buf818  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_187(c_void_p(buf821.data_ptr()), c_void_p(buf817.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(buf820.data_ptr()))
    del addmm_1
    del tanh
    buf822 = buf812; del buf812  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf821, (128, 8192), (8192, 1), 0), permute_1032, out=buf822)
    del permute_1032
    buf823 = empty((8192, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf821, (8192, 128), (1, 8192), 0), view_20, out=buf823)
    del view_20
    buf824 = empty((1, 8192), device='cpu', dtype=torch.float32)
    buf825 = buf814; del buf814  # reuse
    buf826 = buf813; del buf813  # reuse
    buf827 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf828 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf829 = buf817; del buf817  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_188(c_void_p(buf829.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_73.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf825.data_ptr()), c_void_p(buf826.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf828.data_ptr()))
    del buf821
    del div_73
    del mul_2
    del primals_10
    buf830 = buf822; del buf822  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf829, (128, 2048), (2048, 1), 0), permute_1036, out=buf830)
    del permute_1036
    buf831 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf829, (2048, 128), (1, 2048), 0), view_18, out=buf831)
    del view_18
    buf832 = empty((1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_sum_189(c_void_p(buf829.data_ptr()), c_void_p(buf832.data_ptr()))
    buf833 = reinterpret_tensor(buf809, (16, 128, 128), (16384, 128, 1), 0); del buf809  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1041, reinterpret_tensor(buf830, (16, 128, 128), (128, 2048, 1), 0), out=buf833)
    del permute_1041
    buf834 = reinterpret_tensor(buf806, (16, 128, 128), (16384, 128, 1), 0); del buf806  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf830, (16, 128, 128), (128, 2048, 1), 0), permute_1042, out=buf834)
    del permute_1042
    buf835 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf836 = reinterpret_tensor(buf834, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf834  # reuse
    cpp_fused__softmax_backward_data_nll_loss_forward_where_190(c_void_p(buf836.data_ptr()), c_void_p(alias_97.data_ptr()), c_void_p(slice_4.data_ptr()), c_void_p(buf835.data_ptr()))
    del alias_97
    del slice_4
    buf837 = reinterpret_tensor(buf830, (16, 128, 128), (16384, 128, 1), 0); del buf830  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1043, reinterpret_tensor(buf836, (16, 128, 128), (16384, 128, 1), 0), out=buf837)
    del permute_1043
    buf838 = reinterpret_tensor(buf810, (16, 128, 128), (16384, 128, 1), 0); del buf810  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf836, (16, 128, 128), (16384, 128, 1), 0), permute_1044, out=buf838)
    del permute_1044
    buf839 = reinterpret_tensor(buf836, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf836  # reuse
    cpp_fused_clone_191(c_void_p(tangents_4.data_ptr()), c_void_p(buf833.data_ptr()), c_void_p(buf839.data_ptr()))
    del tangents_4
    buf840 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf839, (2048, 128), (1, 2048), 0), view_2, out=buf840)
    buf841 = reinterpret_tensor(buf833, (128, 2048), (2048, 1), 0); del buf833  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf839, (128, 2048), (2048, 1), 0), permute_1051, out=buf841)
    del permute_1051
    buf842 = buf839; del buf839  # reuse
    cpp_fused_clone_192(c_void_p(tangents_3.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf842.data_ptr()))
    del tangents_3
    buf843 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf842, (2048, 128), (1, 2048), 0), view_2, out=buf843)
    buf844 = reinterpret_tensor(buf837, (128, 2048), (2048, 1), 0); del buf837  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf842, (128, 2048), (2048, 1), 0), permute_1055, out=buf844)
    del permute_1055
    buf845 = reinterpret_tensor(buf842, (128, 2048), (2048, 1), 0); del buf842  # reuse
    cpp_fused_view_193(c_void_p(buf838.data_ptr()), c_void_p(buf845.data_ptr()))
    buf846 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf845, (2048, 128), (1, 2048), 0), view_2, out=buf846)
    del view_2
    buf847 = reinterpret_tensor(buf838, (128, 2048), (2048, 1), 0); del buf838  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf845, permute_1059, out=buf847)
    del permute_1059
    buf848 = buf826; del buf826  # reuse
    buf849 = buf825; del buf825  # reuse
    buf850 = reinterpret_tensor(buf835, (2048, ), (1, ), 0); del buf835  # reuse
    buf851 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf852 = buf829; del buf829  # reuse
    buf858 = reinterpret_tensor(buf845, (1, 128, 2048), (262144, 2048, 1), 0); del buf845  # reuse
    buf853 = empty((2048, 2048), device='cpu', dtype=torch.float32)
    buf854 = buf852; del buf852  # reuse
    cpp_fused_add_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_194(c_void_p(buf854.data_ptr()), c_void_p(buf841.data_ptr()), c_void_p(buf844.data_ptr()), c_void_p(buf847.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_74.data_ptr()), c_void_p(view.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(buf850.data_ptr()), c_void_p(buf851.data_ptr()), c_void_p(buf858.data_ptr()), c_void_p(buf853.data_ptr()))
    del buf841
    del buf844
    del buf847
    del buf848
    del buf849
    del div_74
    del mul
    del primals_3
    aten.index_put_(buf853, [view_1], buf854, True)
    del buf854
    del view_1
    buf857 = empty((50257, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_195(c_void_p(buf857.data_ptr()))
    aten.index_put_(buf857, [view], buf858, True)
    del buf858
    del view
    return (buf857, buf853, buf850, buf851, reinterpret_tensor(buf846, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf843, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf840, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf831, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf832, (2048, ), (1, ), 0), buf827, buf828, reinterpret_tensor(buf823, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf824, (8192, ), (1, ), 0), reinterpret_tensor(buf819, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf820, (2048, ), (1, ), 0), buf815, buf816, reinterpret_tensor(buf811, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf808, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf805, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf796, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf797, (2048, ), (1, ), 0), buf792, buf793, reinterpret_tensor(buf788, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf789, (8192, ), (1, ), 0), reinterpret_tensor(buf784, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf785, (2048, ), (1, ), 0), buf780, buf781, reinterpret_tensor(buf776, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf773, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf770, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf761, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf762, (2048, ), (1, ), 0), buf757, buf758, reinterpret_tensor(buf753, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf754, (8192, ), (1, ), 0), reinterpret_tensor(buf749, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf750, (2048, ), (1, ), 0), buf745, buf746, reinterpret_tensor(buf741, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf738, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf735, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf726, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf727, (2048, ), (1, ), 0), buf722, buf723, reinterpret_tensor(buf718, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf719, (8192, ), (1, ), 0), reinterpret_tensor(buf714, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf715, (2048, ), (1, ), 0), buf710, buf711, reinterpret_tensor(buf706, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf703, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf700, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf691, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf692, (2048, ), (1, ), 0), buf687, buf688, reinterpret_tensor(buf683, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf684, (8192, ), (1, ), 0), reinterpret_tensor(buf679, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf680, (2048, ), (1, ), 0), buf675, buf676, reinterpret_tensor(buf671, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf668, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf665, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf656, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf657, (2048, ), (1, ), 0), buf652, buf653, reinterpret_tensor(buf648, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf649, (8192, ), (1, ), 0), reinterpret_tensor(buf644, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf645, (2048, ), (1, ), 0), buf640, buf641, reinterpret_tensor(buf636, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf633, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf630, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf621, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf622, (2048, ), (1, ), 0), buf617, buf618, reinterpret_tensor(buf613, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf614, (8192, ), (1, ), 0), reinterpret_tensor(buf609, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf610, (2048, ), (1, ), 0), buf605, buf606, reinterpret_tensor(buf601, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf598, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf595, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf586, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf587, (2048, ), (1, ), 0), buf582, buf583, reinterpret_tensor(buf578, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf579, (8192, ), (1, ), 0), reinterpret_tensor(buf574, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf575, (2048, ), (1, ), 0), buf570, buf571, reinterpret_tensor(buf566, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf563, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf560, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf551, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf552, (2048, ), (1, ), 0), buf547, buf548, reinterpret_tensor(buf543, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf544, (8192, ), (1, ), 0), reinterpret_tensor(buf539, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf540, (2048, ), (1, ), 0), buf535, buf536, reinterpret_tensor(buf531, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf528, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf525, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf516, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf517, (2048, ), (1, ), 0), buf512, buf513, reinterpret_tensor(buf508, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf509, (8192, ), (1, ), 0), reinterpret_tensor(buf504, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf505, (2048, ), (1, ), 0), buf500, buf501, reinterpret_tensor(buf496, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf493, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf490, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf481, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf482, (2048, ), (1, ), 0), buf477, buf478, reinterpret_tensor(buf473, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf474, (8192, ), (1, ), 0), reinterpret_tensor(buf469, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf470, (2048, ), (1, ), 0), buf465, buf466, reinterpret_tensor(buf461, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf458, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf455, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf446, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf447, (2048, ), (1, ), 0), buf442, buf443, reinterpret_tensor(buf438, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf439, (8192, ), (1, ), 0), reinterpret_tensor(buf434, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf435, (2048, ), (1, ), 0), buf430, buf431, reinterpret_tensor(buf426, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf423, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf420, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf411, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf412, (2048, ), (1, ), 0), buf407, buf408, reinterpret_tensor(buf403, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf404, (8192, ), (1, ), 0), reinterpret_tensor(buf399, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf400, (2048, ), (1, ), 0), buf395, buf396, reinterpret_tensor(buf391, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf388, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf385, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf376, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf377, (2048, ), (1, ), 0), buf372, buf373, reinterpret_tensor(buf368, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf369, (8192, ), (1, ), 0), reinterpret_tensor(buf364, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf365, (2048, ), (1, ), 0), buf360, buf361, reinterpret_tensor(buf356, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf353, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf350, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf341, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf342, (2048, ), (1, ), 0), buf337, buf338, reinterpret_tensor(buf333, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf334, (8192, ), (1, ), 0), reinterpret_tensor(buf329, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf330, (2048, ), (1, ), 0), buf325, buf326, reinterpret_tensor(buf321, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf318, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf315, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf306, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf307, (2048, ), (1, ), 0), buf302, buf303, reinterpret_tensor(buf298, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf299, (8192, ), (1, ), 0), reinterpret_tensor(buf294, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf295, (2048, ), (1, ), 0), buf290, buf291, reinterpret_tensor(buf286, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf283, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf280, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf271, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf272, (2048, ), (1, ), 0), buf267, buf268, reinterpret_tensor(buf263, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf264, (8192, ), (1, ), 0), reinterpret_tensor(buf259, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf260, (2048, ), (1, ), 0), buf255, buf256, reinterpret_tensor(buf251, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf248, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf245, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf236, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf237, (2048, ), (1, ), 0), buf232, buf233, reinterpret_tensor(buf228, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf229, (8192, ), (1, ), 0), reinterpret_tensor(buf224, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf225, (2048, ), (1, ), 0), buf220, buf221, reinterpret_tensor(buf216, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf213, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf210, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf201, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf202, (2048, ), (1, ), 0), buf197, buf198, reinterpret_tensor(buf193, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf194, (8192, ), (1, ), 0), reinterpret_tensor(buf189, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf190, (2048, ), (1, ), 0), buf185, buf186, reinterpret_tensor(buf181, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf178, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf175, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf166, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf167, (2048, ), (1, ), 0), buf162, buf163, reinterpret_tensor(buf158, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf159, (8192, ), (1, ), 0), reinterpret_tensor(buf154, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf155, (2048, ), (1, ), 0), buf150, buf151, reinterpret_tensor(buf146, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf143, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf140, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf131, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf132, (2048, ), (1, ), 0), buf127, buf128, reinterpret_tensor(buf123, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf124, (8192, ), (1, ), 0), reinterpret_tensor(buf119, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf120, (2048, ), (1, ), 0), buf115, buf116, reinterpret_tensor(buf111, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf108, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf105, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf96, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf97, (2048, ), (1, ), 0), buf92, buf93, reinterpret_tensor(buf88, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf89, (8192, ), (1, ), 0), reinterpret_tensor(buf84, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf85, (2048, ), (1, ), 0), buf80, buf81, reinterpret_tensor(buf76, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf73, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf70, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf61, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf62, (2048, ), (1, ), 0), buf57, buf58, reinterpret_tensor(buf53, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf54, (8192, ), (1, ), 0), reinterpret_tensor(buf49, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf50, (2048, ), (1, ), 0), buf45, buf46, reinterpret_tensor(buf41, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf38, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf35, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf26, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf27, (2048, ), (1, ), 0), buf22, buf23, reinterpret_tensor(buf18, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf19, (8192, ), (1, ), 0), reinterpret_tensor(buf14, (2048, 8192), (8192, 1), 0), reinterpret_tensor(buf15, (2048, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf6, (50257, 2048), (2048, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    view = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    view_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    mul = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_4 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_18 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_2 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_8 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_8 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_40 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_10 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_12 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_62 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_18 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_24 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_16 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_84 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_26 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_32 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_20 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_106 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_34 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_112 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_24 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_128 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_48 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_28 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_150 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_50 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_156 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_32 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_172 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_58 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_64 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_36 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_194 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_200 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_40 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_216 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_74 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_222 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_44 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_238 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_82 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_244 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_48 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_260 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_90 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_264 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_96 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_266 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_52 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_282 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_98 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_284 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_37 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_12 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_286 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_104 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_288 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_56 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_304 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_106 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_306 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_13 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_308 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_112 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_310 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_60 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_326 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_114 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_328 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_43 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_14 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_330 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_332 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_64 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_348 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_122 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_350 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_15 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_352 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_128 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_354 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_68 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_370 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_130 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_372 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_49 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_16 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_374 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_136 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_376 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_72 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_392 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_138 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_394 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_17 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_396 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_144 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_398 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_76 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_414 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_146 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_416 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_55 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_18 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_418 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_152 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_420 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_80 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_436 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_154 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_438 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_19 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_440 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_160 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_442 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_84 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_458 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_162 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_460 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_61 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_20 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_462 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_168 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_464 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_88 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_480 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_170 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_482 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_21 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_484 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_176 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_486 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_92 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_502 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_178 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_504 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_67 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_22 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_506 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_184 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_508 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    slice_96 = rand_strided((1, 1, 128, 128), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    view_524 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_186 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_526 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    tanh_23 = rand_strided((1, 128, 8192), (1048576, 8192, 1), device='cpu', dtype=torch.float32)
    view_528 = rand_strided((128, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.float32)
    view_531 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    sub_74 = rand_strided((127, 50257), (50257, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((50257, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_269 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_282 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_51 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_292 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_296 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_300 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_302 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_306 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_316 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_53 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_317 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_318 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_325 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_329 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_333 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_339 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_343 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_349 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_55 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_351 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_362 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_366 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_372 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_57 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_384 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_391 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_395 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_399 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_401 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_405 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_409 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_59 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_416 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_417 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_424 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_428 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_432 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_442 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_447 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_448 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_61 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_449 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_450 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_457 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_461 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_465 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_38 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_467 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_480 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_481 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_63 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_482 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_483 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_490 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_494 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_498 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_500 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_504 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_508 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_513 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_514 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_65 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_515 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_516 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_523 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_527 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_531 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_533 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_537 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_541 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_546 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_547 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_67 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_548 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_549 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_556 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_560 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_564 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_44 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_566 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_570 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_574 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_579 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_580 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_69 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_581 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_582 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_589 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_593 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_597 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_599 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_603 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_607 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_612 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_613 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_71 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_614 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_615 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_622 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_626 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_630 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_632 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_636 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_640 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_645 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_646 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_73 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_647 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_648 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_655 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_659 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_663 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_50 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_665 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_669 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_673 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_678 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_679 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_75 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_680 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_681 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_688 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_692 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_696 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_698 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_702 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_53 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_706 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_711 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_712 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_77 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_713 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_714 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_721 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_725 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_729 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_731 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_735 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_739 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_744 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_745 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_79 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_746 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_747 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_754 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_758 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_762 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_56 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_764 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_768 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_772 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_777 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_778 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_81 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_779 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_780 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_787 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_791 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_795 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_797 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_801 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_59 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_805 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_810 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_811 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_83 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_812 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_813 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_820 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_824 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_828 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_830 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_834 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_838 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_843 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_844 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_85 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_845 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_846 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_853 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_857 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_861 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_62 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_863 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_867 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_871 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_876 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_877 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_87 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_878 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_879 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_886 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_890 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_894 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_64 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_896 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_900 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_65 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_904 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_909 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_910 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_89 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_911 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_912 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_919 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_923 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_927 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_929 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_933 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_67 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_937 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_942 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_943 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_91 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_944 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_945 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_952 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_956 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_960 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_68 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_962 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_966 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_69 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_970 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_975 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_976 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_93 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_977 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_978 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_985 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_989 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_993 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_70 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_995 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_999 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_71 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_1003 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1008 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_1009 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_95 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_1010 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_1011 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_1018 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1022 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1026 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_72 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_1028 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    permute_1032 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_73 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_1036 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1041 = rand_strided((16, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_1042 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    alias_97 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_1043 = rand_strided((16, 128, 128), (128, 1, 2048), device='cpu', dtype=torch.float32)
    permute_1044 = rand_strided((16, 128, 128), (128, 2048, 1), device='cpu', dtype=torch.float32)
    permute_1051 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1055 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1059 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    div_74 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 50257), (6432896, 50257, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_16 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_19 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_22 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_25 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_26 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_27 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_28 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_29 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_30 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_31 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_32 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_33 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_34 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_35 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_36 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_37 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_38 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_39 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_40 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_41 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_42 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_43 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_44 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_45 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_46 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_47 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_48 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_49 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    tangents_50 = rand_strided((1, 16, 128, 128), (262144, 16384, 128, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_10, primals_16, primals_23, primals_29, primals_36, primals_42, primals_49, primals_55, primals_62, primals_68, primals_75, primals_81, primals_88, primals_94, primals_101, primals_107, primals_114, primals_120, primals_127, primals_133, primals_140, primals_146, primals_153, primals_159, primals_166, primals_172, primals_179, primals_185, primals_192, primals_198, primals_205, primals_211, primals_218, primals_224, primals_231, primals_237, primals_244, primals_250, primals_257, primals_263, primals_270, primals_276, primals_283, primals_289, primals_296, primals_302, primals_309, primals_315, primals_343, view, view_1, mul, view_2, slice_4, view_18, mul_2, view_20, addmm_1, tanh, view_22, mul_8, view_24, slice_8, view_40, mul_10, view_42, addmm_4, tanh_1, view_44, mul_16, view_46, slice_12, view_62, mul_18, view_64, addmm_7, tanh_2, view_66, mul_24, view_68, slice_16, view_84, mul_26, view_86, addmm_10, tanh_3, view_88, mul_32, view_90, slice_20, view_106, mul_34, view_108, addmm_13, tanh_4, view_110, mul_40, view_112, slice_24, view_128, mul_42, view_130, addmm_16, tanh_5, view_132, mul_48, view_134, slice_28, view_150, mul_50, view_152, addmm_19, tanh_6, view_154, mul_56, view_156, slice_32, view_172, mul_58, view_174, addmm_22, tanh_7, view_176, mul_64, view_178, slice_36, view_194, mul_66, view_196, addmm_25, tanh_8, view_198, mul_72, view_200, slice_40, view_216, mul_74, view_218, addmm_28, tanh_9, view_220, mul_80, view_222, slice_44, view_238, mul_82, view_240, addmm_31, tanh_10, view_242, mul_88, view_244, slice_48, view_260, mul_90, view_262, addmm_34, tanh_11, view_264, mul_96, view_266, slice_52, view_282, mul_98, view_284, addmm_37, tanh_12, view_286, mul_104, view_288, slice_56, view_304, mul_106, view_306, addmm_40, tanh_13, view_308, mul_112, view_310, slice_60, view_326, mul_114, view_328, addmm_43, tanh_14, view_330, mul_120, view_332, slice_64, view_348, mul_122, view_350, addmm_46, tanh_15, view_352, mul_128, view_354, slice_68, view_370, mul_130, view_372, addmm_49, tanh_16, view_374, mul_136, view_376, slice_72, view_392, mul_138, view_394, addmm_52, tanh_17, view_396, mul_144, view_398, slice_76, view_414, mul_146, view_416, addmm_55, tanh_18, view_418, mul_152, view_420, slice_80, view_436, mul_154, view_438, addmm_58, tanh_19, view_440, mul_160, view_442, slice_84, view_458, mul_162, view_460, addmm_61, tanh_20, view_462, mul_168, view_464, slice_88, view_480, mul_170, view_482, addmm_64, tanh_21, view_484, mul_176, view_486, slice_92, view_502, mul_178, view_504, addmm_67, tanh_22, view_506, mul_184, view_508, slice_96, view_524, mul_186, view_526, addmm_70, tanh_23, view_528, mul_192, view_531, sub_74, convert_element_type, permute_267, div_26, permute_269, permute_273, div_27, permute_277, permute_282, permute_283, alias_51, permute_284, permute_285, permute_292, permute_296, permute_300, div_28, permute_302, permute_306, div_29, permute_310, permute_315, permute_316, alias_53, permute_317, permute_318, permute_325, permute_329, permute_333, div_30, permute_335, permute_339, div_31, permute_343, permute_348, permute_349, alias_55, permute_350, permute_351, permute_358, permute_362, permute_366, div_32, permute_368, permute_372, div_33, permute_376, permute_381, permute_382, alias_57, permute_383, permute_384, permute_391, permute_395, permute_399, div_34, permute_401, permute_405, div_35, permute_409, permute_414, permute_415, alias_59, permute_416, permute_417, permute_424, permute_428, permute_432, div_36, permute_434, permute_438, div_37, permute_442, permute_447, permute_448, alias_61, permute_449, permute_450, permute_457, permute_461, permute_465, div_38, permute_467, permute_471, div_39, permute_475, permute_480, permute_481, alias_63, permute_482, permute_483, permute_490, permute_494, permute_498, div_40, permute_500, permute_504, div_41, permute_508, permute_513, permute_514, alias_65, permute_515, permute_516, permute_523, permute_527, permute_531, div_42, permute_533, permute_537, div_43, permute_541, permute_546, permute_547, alias_67, permute_548, permute_549, permute_556, permute_560, permute_564, div_44, permute_566, permute_570, div_45, permute_574, permute_579, permute_580, alias_69, permute_581, permute_582, permute_589, permute_593, permute_597, div_46, permute_599, permute_603, div_47, permute_607, permute_612, permute_613, alias_71, permute_614, permute_615, permute_622, permute_626, permute_630, div_48, permute_632, permute_636, div_49, permute_640, permute_645, permute_646, alias_73, permute_647, permute_648, permute_655, permute_659, permute_663, div_50, permute_665, permute_669, div_51, permute_673, permute_678, permute_679, alias_75, permute_680, permute_681, permute_688, permute_692, permute_696, div_52, permute_698, permute_702, div_53, permute_706, permute_711, permute_712, alias_77, permute_713, permute_714, permute_721, permute_725, permute_729, div_54, permute_731, permute_735, div_55, permute_739, permute_744, permute_745, alias_79, permute_746, permute_747, permute_754, permute_758, permute_762, div_56, permute_764, permute_768, div_57, permute_772, permute_777, permute_778, alias_81, permute_779, permute_780, permute_787, permute_791, permute_795, div_58, permute_797, permute_801, div_59, permute_805, permute_810, permute_811, alias_83, permute_812, permute_813, permute_820, permute_824, permute_828, div_60, permute_830, permute_834, div_61, permute_838, permute_843, permute_844, alias_85, permute_845, permute_846, permute_853, permute_857, permute_861, div_62, permute_863, permute_867, div_63, permute_871, permute_876, permute_877, alias_87, permute_878, permute_879, permute_886, permute_890, permute_894, div_64, permute_896, permute_900, div_65, permute_904, permute_909, permute_910, alias_89, permute_911, permute_912, permute_919, permute_923, permute_927, div_66, permute_929, permute_933, div_67, permute_937, permute_942, permute_943, alias_91, permute_944, permute_945, permute_952, permute_956, permute_960, div_68, permute_962, permute_966, div_69, permute_970, permute_975, permute_976, alias_93, permute_977, permute_978, permute_985, permute_989, permute_993, div_70, permute_995, permute_999, div_71, permute_1003, permute_1008, permute_1009, alias_95, permute_1010, permute_1011, permute_1018, permute_1022, permute_1026, div_72, permute_1028, permute_1032, div_73, permute_1036, permute_1041, permute_1042, alias_97, permute_1043, permute_1044, permute_1051, permute_1055, permute_1059, div_74, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPTNeoForCausalLM', benchmark_compiled_module)
