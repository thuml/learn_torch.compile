
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(15627264L); x0+=static_cast<long>(8L))
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


cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(30520L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30522L*x0)));
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
                    for(long x1=static_cast<long>(30520L); x1<static_cast<long>(30522L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (30522L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(30520L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (30522L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30522L*x0)));
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (30522L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (30522L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(30520L); x1<static_cast<long>(30522L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x1 + (30522L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1 + (30522L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp11 = in_ptr5[static_cast<long>(x1 + (30522L*x0))];
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
                    in_out_ptr0[static_cast<long>(x1 + (30522L*x0))] = tmp16;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(30520L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (30522L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(30520L); x0<static_cast<long>(30522L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (30522L*x1))];
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


cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       float* out_ptr5,
                       float* out_ptr6)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
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
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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


cpp_fused_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_12 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_14 = async_compile.cpp('''
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


cpp_fused_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_20 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_22 = async_compile.cpp('''
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


cpp_fused_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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


cpp_fused_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_36 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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


cpp_fused_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_44 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_46 = async_compile.cpp('''
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


cpp_fused_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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


cpp_fused_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_59 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_60 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_62 = async_compile.cpp('''
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


cpp_fused_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_68 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_70 = async_compile.cpp('''
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


cpp_fused_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_75 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_76 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_78 = async_compile.cpp('''
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


cpp_fused_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_84 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_86 = async_compile.cpp('''
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


cpp_fused_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_91 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_92 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_sum_94 = async_compile.cpp('''
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


cpp_fused_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_99 = async_compile.cpp('''
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
                       const long* in_ptr8,
                       const long* in_ptr9,
                       const long* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = in_ptr3[static_cast<long>(x0)];
                auto tmp7 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                auto tmp8 = c10::convert<float>(tmp7);
                auto tmp9 = static_cast<float>(1.1111111111111112);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp11 = decltype(tmp6)(tmp6 * tmp10);
                in_out_ptr0[static_cast<long>(x0)] = tmp11;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
                    auto tmp17 = in_ptr8[static_cast<long>(x0)];
                    auto tmp24 = in_ptr9[static_cast<long>(x0)];
                    auto tmp28 = in_ptr10[static_cast<long>(x0)];
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
                    auto tmp18 = static_cast<int>(-1);
                    auto tmp19 = tmp17 == tmp18;
                    auto tmp20 = static_cast<float>(0.0);
                    auto tmp21 = to_float_mask(tmp19);
                    auto tmp22 = at::vec::Vectorized<float>(tmp20);
                    auto tmp23 = decltype(tmp22)::blendv(tmp16, tmp22, tmp21);
                    auto tmp25 = tmp24 == tmp18;
                    auto tmp26 = to_float_mask(tmp25);
                    auto tmp27 = decltype(tmp22)::blendv(tmp16, tmp22, tmp26);
                    auto tmp29 = static_cast<int>(0);
                    auto tmp30 = tmp28 == tmp29;
                    auto tmp31 = to_float_mask(tmp30);
                    auto tmp32 = decltype(tmp22)::blendv(tmp16, tmp22, tmp31);
                    tmp23.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    tmp27.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    tmp32.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr7 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr9 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_embedding_dense_backward_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(23440896L); x0+=static_cast<long>(8L))
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
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, primals_207, expand, slice_4, mul_1, getitem_3, view, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, addmm_72, mul_90, view_266, sub_40, convert_element_type, permute_134, div_26, permute_138, div_27, permute_142, permute_146, div_28, permute_150, permute_162, permute_167, permute_171, div_30, permute_175, permute_179, div_31, permute_183, permute_195, permute_200, permute_204, div_33, permute_208, permute_212, div_34, permute_216, permute_228, permute_233, permute_237, div_36, permute_241, permute_245, div_37, permute_249, permute_261, permute_266, permute_270, div_39, permute_274, permute_278, div_40, permute_282, permute_294, permute_299, permute_303, div_42, permute_307, permute_311, div_43, permute_315, permute_327, permute_332, permute_336, div_45, permute_340, permute_344, div_46, permute_348, permute_360, permute_365, permute_369, div_48, permute_373, permute_377, div_49, permute_381, permute_393, permute_398, permute_402, div_51, permute_406, permute_410, div_52, permute_414, permute_426, permute_431, permute_435, div_54, permute_439, permute_443, div_55, permute_447, permute_459, permute_464, permute_468, div_57, permute_472, permute_476, div_58, permute_480, permute_492, permute_497, permute_501, div_60, permute_505, permute_509, div_61, permute_513, permute_525, permute_530, permute_534, div_63, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_206, (1, 512), (512, 1))
    assert_size_stride(primals_207, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_4, (1, 512), (512, 1))
    assert_size_stride(mul_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(getitem_149, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_67, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_68, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_23, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_69, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_70, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_16, (512, 768), (768, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(getitem_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(getitem_147, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_61, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_62, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_21, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_63, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_64, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(getitem_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_40, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_42, (512, 3072), (3072, 1))
    assert_size_stride(getitem_21, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_15, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_44, (512, 768), (768, 1))
    assert_size_stride(getitem_145, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_55, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_56, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_19, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_57, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_58, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_60, (512, 768), (768, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_62, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_64, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_22, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(getitem_143, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_49, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_50, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_17, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_51, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_52, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_82, (512, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_86, (512, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_29, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_88, (512, 768), (768, 1))
    assert_size_stride(getitem_141, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_43, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_44, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_15, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_45, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_46, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(getitem_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_106, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_108, (512, 3072), (3072, 1))
    assert_size_stride(getitem_51, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_110, (512, 768), (768, 1))
    assert_size_stride(getitem_139, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_37, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_38, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_13, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_39, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_40, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(getitem_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_38, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_128, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_130, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_132, (512, 768), (768, 1))
    assert_size_stride(getitem_137, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_31, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_32, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_11, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_33, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_34, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_148, (512, 768), (768, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_45, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_150, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_152, (512, 3072), (3072, 1))
    assert_size_stride(getitem_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_50, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_154, (512, 768), (768, 1))
    assert_size_stride(getitem_135, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_25, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_26, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_9, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_27, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_28, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_170, (512, 768), (768, 1))
    assert_size_stride(getitem_77, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_52, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_172, (512, 768), (768, 1))
    assert_size_stride(addmm_46, (512, 3072), (3072, 1))
    assert_size_stride(view_174, (512, 3072), (3072, 1))
    assert_size_stride(getitem_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(getitem_133, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_19, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_20, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_7, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_21, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_22, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(getitem_87, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_59, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_52, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(getitem_91, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_64, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(getitem_131, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_13, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_14, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_5, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_15, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_16, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_214, (512, 768), (768, 1))
    assert_size_stride(getitem_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_66, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_58, (512, 3072), (3072, 1))
    assert_size_stride(view_218, (512, 3072), (3072, 1))
    assert_size_stride(getitem_101, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_220, (512, 768), (768, 1))
    assert_size_stride(getitem_129, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_7, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_8, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_3, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_9, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_10, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_236, (512, 768), (768, 1))
    assert_size_stride(getitem_107, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_238, (512, 768), (768, 1))
    assert_size_stride(addmm_64, (512, 3072), (3072, 1))
    assert_size_stride(view_240, (512, 3072), (3072, 1))
    assert_size_stride(getitem_111, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_78, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_242, (512, 768), (768, 1))
    assert_size_stride(getitem_127, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_1, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_2, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_1, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_3, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_4, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_258, (512, 768), (768, 1))
    assert_size_stride(getitem_117, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_80, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_260, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 3072), (3072, 1))
    assert_size_stride(view_262, (512, 3072), (3072, 1))
    assert_size_stride(getitem_121, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_85, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_264, (512, 768), (768, 1))
    assert_size_stride(addmm_72, (512, 768), (768, 1))
    assert_size_stride(mul_90, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_266, (512, 768), (768, 1))
    assert_size_stride(sub_40, (512, 30522), (30522, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_134, (30522, 768), (768, 1))
    assert_size_stride(div_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_142, (768, 3072), (3072, 1))
    assert_size_stride(permute_146, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_150, (768, 768), (768, 1))
    assert_size_stride(permute_162, (768, 768), (768, 1))
    assert_size_stride(permute_167, (768, 768), (768, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_195, (768, 768), (768, 1))
    assert_size_stride(permute_200, (768, 768), (768, 1))
    assert_size_stride(permute_204, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_228, (768, 768), (768, 1))
    assert_size_stride(permute_233, (768, 768), (768, 1))
    assert_size_stride(permute_237, (768, 768), (768, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_241, (768, 3072), (3072, 1))
    assert_size_stride(permute_245, (3072, 768), (768, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_249, (768, 768), (768, 1))
    assert_size_stride(permute_261, (768, 768), (768, 1))
    assert_size_stride(permute_266, (768, 768), (768, 1))
    assert_size_stride(permute_270, (768, 768), (768, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_274, (768, 3072), (3072, 1))
    assert_size_stride(permute_278, (3072, 768), (768, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_282, (768, 768), (768, 1))
    assert_size_stride(permute_294, (768, 768), (768, 1))
    assert_size_stride(permute_299, (768, 768), (768, 1))
    assert_size_stride(permute_303, (768, 768), (768, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_307, (768, 3072), (3072, 1))
    assert_size_stride(permute_311, (3072, 768), (768, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_315, (768, 768), (768, 1))
    assert_size_stride(permute_327, (768, 768), (768, 1))
    assert_size_stride(permute_332, (768, 768), (768, 1))
    assert_size_stride(permute_336, (768, 768), (768, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_340, (768, 3072), (3072, 1))
    assert_size_stride(permute_344, (3072, 768), (768, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_348, (768, 768), (768, 1))
    assert_size_stride(permute_360, (768, 768), (768, 1))
    assert_size_stride(permute_365, (768, 768), (768, 1))
    assert_size_stride(permute_369, (768, 768), (768, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_373, (768, 3072), (3072, 1))
    assert_size_stride(permute_377, (3072, 768), (768, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_381, (768, 768), (768, 1))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_398, (768, 768), (768, 1))
    assert_size_stride(permute_402, (768, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_406, (768, 3072), (3072, 1))
    assert_size_stride(permute_410, (3072, 768), (768, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_414, (768, 768), (768, 1))
    assert_size_stride(permute_426, (768, 768), (768, 1))
    assert_size_stride(permute_431, (768, 768), (768, 1))
    assert_size_stride(permute_435, (768, 768), (768, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_439, (768, 3072), (3072, 1))
    assert_size_stride(permute_443, (3072, 768), (768, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_447, (768, 768), (768, 1))
    assert_size_stride(permute_459, (768, 768), (768, 1))
    assert_size_stride(permute_464, (768, 768), (768, 1))
    assert_size_stride(permute_468, (768, 768), (768, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_472, (768, 3072), (3072, 1))
    assert_size_stride(permute_476, (3072, 768), (768, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_480, (768, 768), (768, 1))
    assert_size_stride(permute_492, (768, 768), (768, 1))
    assert_size_stride(permute_497, (768, 768), (768, 1))
    assert_size_stride(permute_501, (768, 768), (768, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_505, (768, 3072), (3072, 1))
    assert_size_stride(permute_509, (3072, 768), (768, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_513, (768, 768), (768, 1))
    assert_size_stride(permute_525, (768, 768), (768, 1))
    assert_size_stride(permute_530, (768, 768), (768, 1))
    assert_size_stride(permute_534, (768, 768), (768, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 30522), (15627264, 30522, 1))
    buf0 = empty((512, 30522), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_207.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((512, 30522), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 512, 30522), (15627264, 30522, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_40.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del convert_element_type
    del primals_207
    del sub_40
    del tangents_1
    del tangents_2
    buf6 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (512, 30522), (30522, 1), 0), permute_134, out=buf6)
    del permute_134
    buf7 = empty((30522, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (30522, 512), (1, 30522), 0), view_266, out=buf7)
    del view_266
    buf8 = empty((1, 30522), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf4, (1, 512, 1), (512, 1, 512), 0); del buf4  # reuse
    buf10 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf11 = empty((768, ), device='cpu', dtype=torch.float32)
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf6, (1, 512, 768), (393216, 768, 1), 0); del buf6  # reuse
    cpp_fused_gelu_gelu_backward_native_layer_norm_backward_sum_2(c_void_p(buf13.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(addmm_72.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del addmm_72
    del buf5
    del div_26
    del mul_90
    del primals_200
    buf14 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_138, out=buf14)
    del permute_138
    buf15 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (768, 512), (1, 768), 0), view_264, out=buf15)
    del view_264
    buf16 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf17 = buf9; del buf9  # reuse
    buf18 = buf10; del buf10  # reuse
    buf19 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf20 = empty((768, ), device='cpu', dtype=torch.float32)
    buf21 = empty((768, ), device='cpu', dtype=torch.float32)
    buf22 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(mul_85.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(getitem_121.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del div_27
    del getitem_121
    del mul_85
    del primals_196
    buf23 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (512, 768), (768, 1), 0), permute_142, out=buf23)
    del permute_142
    buf24 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (768, 512), (1, 768), 0), view_262, out=buf24)
    del view_262
    buf25 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf23, (1, 512, 3072), (1572864, 3072, 1), 0); del buf23  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf26.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(buf25.data_ptr()))
    del addmm_70
    buf27 = reinterpret_tensor(buf22, (512, 768), (768, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0), permute_146, out=buf27)
    del permute_146
    buf28 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (3072, 512), (1, 3072), 0), view_260, out=buf28)
    del view_260
    buf29 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf30 = buf18; del buf18  # reuse
    buf31 = buf17; del buf17  # reuse
    buf32 = reinterpret_tensor(buf14, (1, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
    buf33 = empty((768, ), device='cpu', dtype=torch.float32)
    buf34 = empty((768, ), device='cpu', dtype=torch.float32)
    buf35 = buf13; del buf13  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_5(c_void_p(buf26.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del div_28
    del getitem_117
    del mul_80
    del primals_190
    buf36 = buf27; del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (512, 768), (768, 1), 0), permute_150, out=buf36)
    del permute_150
    buf37 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (768, 512), (1, 768), 0), view_258, out=buf37)
    del view_258
    buf38 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_6(c_void_p(buf35.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf35, (12, 512, 64), (32768, 64, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_1, reinterpret_tensor(buf36, (12, 512, 64), (64, 768, 1), 0), out=buf39)
    del permute_default_1
    buf40 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf36, (12, 512, 64), (64, 768, 1), 0), permute_default_2, out=buf40)
    del permute_default_2
    buf41 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf42 = reinterpret_tensor(buf40, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf40  # reuse
    cpp_fused_7(c_void_p(buf42.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(alias_default_1.data_ptr()), c_void_p(buf41.data_ptr()))
    del alias_default_1
    del getitem_127
    buf43 = reinterpret_tensor(buf36, (12, 64, 512), (32768, 512, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_3, reinterpret_tensor(buf42, (12, 512, 512), (262144, 512, 1), 0), out=buf43)
    del permute_default_3
    buf44 = reinterpret_tensor(buf19, (12, 512, 64), (32768, 64, 1), 0); del buf19  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf42, (12, 512, 512), (262144, 512, 1), 0), permute_default_4, out=buf44)
    del permute_default_4
    buf45 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_8(c_void_p(buf39.data_ptr()), c_void_p(buf45.data_ptr()))
    buf46 = reinterpret_tensor(buf39, (512, 768), (768, 1), 0); del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf45, permute_162, out=buf46)
    del permute_162
    buf47 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (768, 512), (1, 768), 0), view_242, out=buf47)
    buf48 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf49 = reinterpret_tensor(buf43, (512, 768), (1, 512), 0); del buf43  # reuse
    cpp_fused_sum_view_9(c_void_p(buf49.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf48.data_ptr()))
    buf50 = buf45; del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf49, permute_167, out=buf50)
    del permute_167
    buf51 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (768, 512), (512, 1), 0), view_242, out=buf51)
    buf52 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf53 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_10(c_void_p(buf49.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    buf54 = reinterpret_tensor(buf49, (512, 768), (768, 1), 0); del buf49  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf53, permute_171, out=buf54)
    del permute_171
    buf55 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf53, (768, 512), (1, 768), 0), view_242, out=buf55)
    del view_242
    buf56 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf57 = reinterpret_tensor(buf44, (1, 512, 768), (393216, 768, 1), 0); del buf44  # reuse
    buf58 = buf31; del buf31  # reuse
    buf59 = buf30; del buf30  # reuse
    buf60 = buf57; del buf57  # reuse
    buf61 = empty((768, ), device='cpu', dtype=torch.float32)
    buf62 = empty((768, ), device='cpu', dtype=torch.float32)
    buf63 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11(c_void_p(buf60.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(mul_78.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    del div_30
    del getitem_111
    del mul_78
    del primals_180
    buf64 = reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (512, 768), (768, 1), 0), permute_175, out=buf64)
    del permute_175
    buf65 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (768, 512), (1, 768), 0), view_240, out=buf65)
    del view_240
    buf66 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf67 = reinterpret_tensor(buf64, (1, 512, 3072), (1572864, 3072, 1), 0); del buf64  # reuse
    cpp_fused_gelu_gelu_backward_sum_12(c_void_p(buf67.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(addmm_64.data_ptr()), c_void_p(buf66.data_ptr()))
    del addmm_64
    buf68 = reinterpret_tensor(buf63, (512, 768), (768, 1), 0); del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (512, 3072), (3072, 1), 0), permute_179, out=buf68)
    del permute_179
    buf69 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (3072, 512), (1, 3072), 0), view_238, out=buf69)
    del view_238
    buf70 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf71 = buf59; del buf59  # reuse
    buf72 = buf58; del buf58  # reuse
    buf73 = reinterpret_tensor(buf54, (1, 512, 768), (393216, 768, 1), 0); del buf54  # reuse
    buf74 = empty((768, ), device='cpu', dtype=torch.float32)
    buf75 = empty((768, ), device='cpu', dtype=torch.float32)
    buf76 = reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0); del buf53  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13(c_void_p(buf67.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del div_31
    del getitem_107
    del mul_73
    del primals_174
    buf77 = buf68; del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (512, 768), (768, 1), 0), permute_183, out=buf77)
    del permute_183
    buf78 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (768, 512), (1, 768), 0), view_236, out=buf78)
    del view_236
    buf79 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_14(c_void_p(buf76.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = reinterpret_tensor(buf76, (12, 512, 64), (32768, 64, 1), 0); del buf76  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_7, reinterpret_tensor(buf77, (12, 512, 64), (64, 768, 1), 0), out=buf80)
    del permute_default_7
    buf81 = reinterpret_tensor(buf42, (12, 512, 512), (262144, 512, 1), 0); del buf42  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf77, (12, 512, 64), (64, 768, 1), 0), permute_default_8, out=buf81)
    del permute_default_8
    buf82 = buf41; del buf41  # reuse
    buf83 = reinterpret_tensor(buf81, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf81  # reuse
    cpp_fused_15(c_void_p(buf83.data_ptr()), c_void_p(getitem_129.data_ptr()), c_void_p(alias_default_3.data_ptr()), c_void_p(buf82.data_ptr()))
    del alias_default_3
    del getitem_129
    buf84 = reinterpret_tensor(buf77, (12, 64, 512), (32768, 512, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_9, reinterpret_tensor(buf83, (12, 512, 512), (262144, 512, 1), 0), out=buf84)
    del permute_default_9
    buf85 = reinterpret_tensor(buf60, (12, 512, 64), (32768, 64, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf83, (12, 512, 512), (262144, 512, 1), 0), permute_default_10, out=buf85)
    del permute_default_10
    buf86 = buf50; del buf50  # reuse
    cpp_fused_view_16(c_void_p(buf80.data_ptr()), c_void_p(buf86.data_ptr()))
    buf87 = reinterpret_tensor(buf80, (512, 768), (768, 1), 0); del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf86, permute_195, out=buf87)
    del permute_195
    buf88 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (768, 512), (1, 768), 0), view_220, out=buf88)
    buf89 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf90 = reinterpret_tensor(buf84, (512, 768), (1, 512), 0); del buf84  # reuse
    cpp_fused_sum_view_17(c_void_p(buf90.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf89.data_ptr()))
    buf91 = buf86; del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf90, permute_200, out=buf91)
    del permute_200
    buf92 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (768, 512), (512, 1), 0), view_220, out=buf92)
    buf93 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf94 = buf46; del buf46  # reuse
    cpp_fused_sum_view_18(c_void_p(buf90.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    buf95 = reinterpret_tensor(buf90, (512, 768), (768, 1), 0); del buf90  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf94, permute_204, out=buf95)
    del permute_204
    buf96 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (768, 512), (1, 768), 0), view_220, out=buf96)
    del view_220
    buf97 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf98 = reinterpret_tensor(buf85, (1, 512, 768), (393216, 768, 1), 0); del buf85  # reuse
    buf99 = buf72; del buf72  # reuse
    buf100 = buf71; del buf71  # reuse
    buf101 = buf98; del buf98  # reuse
    buf102 = empty((768, ), device='cpu', dtype=torch.float32)
    buf103 = empty((768, ), device='cpu', dtype=torch.float32)
    buf104 = buf32; del buf32  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19(c_void_p(buf101.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(mul_71.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del div_33
    del getitem_101
    del mul_71
    del primals_164
    buf105 = reinterpret_tensor(buf67, (512, 3072), (3072, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (512, 768), (768, 1), 0), permute_208, out=buf105)
    del permute_208
    buf106 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (768, 512), (1, 768), 0), view_218, out=buf106)
    del view_218
    buf107 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf105, (1, 512, 3072), (1572864, 3072, 1), 0); del buf105  # reuse
    cpp_fused_gelu_gelu_backward_sum_20(c_void_p(buf108.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf107.data_ptr()))
    del addmm_58
    buf109 = reinterpret_tensor(buf104, (512, 768), (768, 1), 0); del buf104  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf108, (512, 3072), (3072, 1), 0), permute_212, out=buf109)
    del permute_212
    buf110 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf108, (3072, 512), (1, 3072), 0), view_216, out=buf110)
    del view_216
    buf111 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf112 = buf99; del buf99  # reuse
    buf113 = buf100; del buf100  # reuse
    buf114 = reinterpret_tensor(buf95, (1, 512, 768), (393216, 768, 1), 0); del buf95  # reuse
    buf115 = empty((768, ), device='cpu', dtype=torch.float32)
    buf116 = empty((768, ), device='cpu', dtype=torch.float32)
    buf117 = reinterpret_tensor(buf94, (1, 512, 768), (393216, 768, 1), 0); del buf94  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21(c_void_p(buf108.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del div_34
    del getitem_97
    del mul_66
    del primals_158
    buf118 = buf109; del buf109  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (512, 768), (768, 1), 0), permute_216, out=buf118)
    del permute_216
    buf119 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (768, 512), (1, 768), 0), view_214, out=buf119)
    del view_214
    buf120 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_22(c_void_p(buf117.data_ptr()), c_void_p(buf120.data_ptr()))
    buf121 = reinterpret_tensor(buf117, (12, 512, 64), (32768, 64, 1), 0); del buf117  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_13, reinterpret_tensor(buf118, (12, 512, 64), (64, 768, 1), 0), out=buf121)
    del permute_default_13
    buf122 = reinterpret_tensor(buf83, (12, 512, 512), (262144, 512, 1), 0); del buf83  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf118, (12, 512, 64), (64, 768, 1), 0), permute_default_14, out=buf122)
    del permute_default_14
    buf123 = buf82; del buf82  # reuse
    buf124 = reinterpret_tensor(buf122, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf122  # reuse
    cpp_fused_23(c_void_p(buf124.data_ptr()), c_void_p(getitem_131.data_ptr()), c_void_p(alias_default_5.data_ptr()), c_void_p(buf123.data_ptr()))
    del alias_default_5
    del getitem_131
    buf125 = reinterpret_tensor(buf118, (12, 64, 512), (32768, 512, 1), 0); del buf118  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_15, reinterpret_tensor(buf124, (12, 512, 512), (262144, 512, 1), 0), out=buf125)
    del permute_default_15
    buf126 = reinterpret_tensor(buf101, (12, 512, 64), (32768, 64, 1), 0); del buf101  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf124, (12, 512, 512), (262144, 512, 1), 0), permute_default_16, out=buf126)
    del permute_default_16
    buf127 = buf91; del buf91  # reuse
    cpp_fused_view_24(c_void_p(buf121.data_ptr()), c_void_p(buf127.data_ptr()))
    buf128 = reinterpret_tensor(buf121, (512, 768), (768, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf127, permute_228, out=buf128)
    del permute_228
    buf129 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (768, 512), (1, 768), 0), view_198, out=buf129)
    buf130 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf131 = reinterpret_tensor(buf125, (512, 768), (1, 512), 0); del buf125  # reuse
    cpp_fused_sum_view_25(c_void_p(buf131.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf130.data_ptr()))
    buf132 = buf127; del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf131, permute_233, out=buf132)
    del permute_233
    buf133 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (768, 512), (512, 1), 0), view_198, out=buf133)
    buf134 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf135 = buf87; del buf87  # reuse
    cpp_fused_sum_view_26(c_void_p(buf131.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = reinterpret_tensor(buf131, (512, 768), (768, 1), 0); del buf131  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf135, permute_237, out=buf136)
    del permute_237
    buf137 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (768, 512), (1, 768), 0), view_198, out=buf137)
    del view_198
    buf138 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf139 = reinterpret_tensor(buf126, (1, 512, 768), (393216, 768, 1), 0); del buf126  # reuse
    buf140 = buf113; del buf113  # reuse
    buf141 = buf112; del buf112  # reuse
    buf142 = buf139; del buf139  # reuse
    buf143 = empty((768, ), device='cpu', dtype=torch.float32)
    buf144 = empty((768, ), device='cpu', dtype=torch.float32)
    buf145 = buf73; del buf73  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27(c_void_p(buf142.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del div_36
    del getitem_91
    del mul_64
    del primals_148
    buf146 = reinterpret_tensor(buf108, (512, 3072), (3072, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (512, 768), (768, 1), 0), permute_241, out=buf146)
    del permute_241
    buf147 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (768, 512), (1, 768), 0), view_196, out=buf147)
    del view_196
    buf148 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf149 = reinterpret_tensor(buf146, (1, 512, 3072), (1572864, 3072, 1), 0); del buf146  # reuse
    cpp_fused_gelu_gelu_backward_sum_28(c_void_p(buf149.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(buf148.data_ptr()))
    del addmm_52
    buf150 = reinterpret_tensor(buf145, (512, 768), (768, 1), 0); del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0), permute_245, out=buf150)
    del permute_245
    buf151 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (3072, 512), (1, 3072), 0), view_194, out=buf151)
    del view_194
    buf152 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf153 = buf141; del buf141  # reuse
    buf154 = buf140; del buf140  # reuse
    buf155 = reinterpret_tensor(buf136, (1, 512, 768), (393216, 768, 1), 0); del buf136  # reuse
    buf156 = empty((768, ), device='cpu', dtype=torch.float32)
    buf157 = empty((768, ), device='cpu', dtype=torch.float32)
    buf158 = reinterpret_tensor(buf135, (1, 512, 768), (393216, 768, 1), 0); del buf135  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_29(c_void_p(buf149.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    del div_37
    del getitem_87
    del mul_59
    del primals_142
    buf159 = buf150; del buf150  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (512, 768), (768, 1), 0), permute_249, out=buf159)
    del permute_249
    buf160 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (768, 512), (1, 768), 0), view_192, out=buf160)
    del view_192
    buf161 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_30(c_void_p(buf158.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = reinterpret_tensor(buf158, (12, 512, 64), (32768, 64, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_19, reinterpret_tensor(buf159, (12, 512, 64), (64, 768, 1), 0), out=buf162)
    del permute_default_19
    buf163 = reinterpret_tensor(buf124, (12, 512, 512), (262144, 512, 1), 0); del buf124  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf159, (12, 512, 64), (64, 768, 1), 0), permute_default_20, out=buf163)
    del permute_default_20
    buf164 = buf123; del buf123  # reuse
    buf165 = reinterpret_tensor(buf163, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf163  # reuse
    cpp_fused_31(c_void_p(buf165.data_ptr()), c_void_p(getitem_133.data_ptr()), c_void_p(alias_default_7.data_ptr()), c_void_p(buf164.data_ptr()))
    del alias_default_7
    del getitem_133
    buf166 = reinterpret_tensor(buf159, (12, 64, 512), (32768, 512, 1), 0); del buf159  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_21, reinterpret_tensor(buf165, (12, 512, 512), (262144, 512, 1), 0), out=buf166)
    del permute_default_21
    buf167 = reinterpret_tensor(buf142, (12, 512, 64), (32768, 64, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf165, (12, 512, 512), (262144, 512, 1), 0), permute_default_22, out=buf167)
    del permute_default_22
    buf168 = buf132; del buf132  # reuse
    cpp_fused_view_32(c_void_p(buf162.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = reinterpret_tensor(buf162, (512, 768), (768, 1), 0); del buf162  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf168, permute_261, out=buf169)
    del permute_261
    buf170 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (768, 512), (1, 768), 0), view_176, out=buf170)
    buf171 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf172 = reinterpret_tensor(buf166, (512, 768), (1, 512), 0); del buf166  # reuse
    cpp_fused_sum_view_33(c_void_p(buf172.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf171.data_ptr()))
    buf173 = buf168; del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf172, permute_266, out=buf173)
    del permute_266
    buf174 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf172, (768, 512), (512, 1), 0), view_176, out=buf174)
    buf175 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf176 = buf128; del buf128  # reuse
    cpp_fused_sum_view_34(c_void_p(buf172.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    buf177 = reinterpret_tensor(buf172, (512, 768), (768, 1), 0); del buf172  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf176, permute_270, out=buf177)
    del permute_270
    buf178 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf176, (768, 512), (1, 768), 0), view_176, out=buf178)
    del view_176
    buf179 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf180 = reinterpret_tensor(buf167, (1, 512, 768), (393216, 768, 1), 0); del buf167  # reuse
    buf181 = buf154; del buf154  # reuse
    buf182 = buf153; del buf153  # reuse
    buf183 = buf180; del buf180  # reuse
    buf184 = empty((768, ), device='cpu', dtype=torch.float32)
    buf185 = empty((768, ), device='cpu', dtype=torch.float32)
    buf186 = buf114; del buf114  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35(c_void_p(buf183.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del div_39
    del getitem_81
    del mul_57
    del primals_132
    buf187 = reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (512, 768), (768, 1), 0), permute_274, out=buf187)
    del permute_274
    buf188 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (768, 512), (1, 768), 0), view_174, out=buf188)
    del view_174
    buf189 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf190 = reinterpret_tensor(buf187, (1, 512, 3072), (1572864, 3072, 1), 0); del buf187  # reuse
    cpp_fused_gelu_gelu_backward_sum_36(c_void_p(buf190.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf189.data_ptr()))
    del addmm_46
    buf191 = reinterpret_tensor(buf186, (512, 768), (768, 1), 0); del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (512, 3072), (3072, 1), 0), permute_278, out=buf191)
    del permute_278
    buf192 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (3072, 512), (1, 3072), 0), view_172, out=buf192)
    del view_172
    buf193 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf194 = buf182; del buf182  # reuse
    buf195 = buf181; del buf181  # reuse
    buf196 = reinterpret_tensor(buf177, (1, 512, 768), (393216, 768, 1), 0); del buf177  # reuse
    buf197 = empty((768, ), device='cpu', dtype=torch.float32)
    buf198 = empty((768, ), device='cpu', dtype=torch.float32)
    buf199 = reinterpret_tensor(buf176, (1, 512, 768), (393216, 768, 1), 0); del buf176  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_37(c_void_p(buf190.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(mul_52.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del div_40
    del getitem_77
    del mul_52
    del primals_126
    buf200 = buf191; del buf191  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (512, 768), (768, 1), 0), permute_282, out=buf200)
    del permute_282
    buf201 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (768, 512), (1, 768), 0), view_170, out=buf201)
    del view_170
    buf202 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_38(c_void_p(buf199.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf199, (12, 512, 64), (32768, 64, 1), 0); del buf199  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_25, reinterpret_tensor(buf200, (12, 512, 64), (64, 768, 1), 0), out=buf203)
    del permute_default_25
    buf204 = reinterpret_tensor(buf165, (12, 512, 512), (262144, 512, 1), 0); del buf165  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf200, (12, 512, 64), (64, 768, 1), 0), permute_default_26, out=buf204)
    del permute_default_26
    buf205 = buf164; del buf164  # reuse
    buf206 = reinterpret_tensor(buf204, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf204  # reuse
    cpp_fused_39(c_void_p(buf206.data_ptr()), c_void_p(getitem_135.data_ptr()), c_void_p(alias_default_9.data_ptr()), c_void_p(buf205.data_ptr()))
    del alias_default_9
    del getitem_135
    buf207 = reinterpret_tensor(buf200, (12, 64, 512), (32768, 512, 1), 0); del buf200  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_27, reinterpret_tensor(buf206, (12, 512, 512), (262144, 512, 1), 0), out=buf207)
    del permute_default_27
    buf208 = reinterpret_tensor(buf183, (12, 512, 64), (32768, 64, 1), 0); del buf183  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf206, (12, 512, 512), (262144, 512, 1), 0), permute_default_28, out=buf208)
    del permute_default_28
    buf209 = buf173; del buf173  # reuse
    cpp_fused_view_40(c_void_p(buf203.data_ptr()), c_void_p(buf209.data_ptr()))
    buf210 = reinterpret_tensor(buf203, (512, 768), (768, 1), 0); del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf209, permute_294, out=buf210)
    del permute_294
    buf211 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (768, 512), (1, 768), 0), view_154, out=buf211)
    buf212 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf213 = reinterpret_tensor(buf207, (512, 768), (1, 512), 0); del buf207  # reuse
    cpp_fused_sum_view_41(c_void_p(buf213.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf212.data_ptr()))
    buf214 = buf209; del buf209  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf213, permute_299, out=buf214)
    del permute_299
    buf215 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (768, 512), (512, 1), 0), view_154, out=buf215)
    buf216 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf217 = buf169; del buf169  # reuse
    cpp_fused_sum_view_42(c_void_p(buf213.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    buf218 = reinterpret_tensor(buf213, (512, 768), (768, 1), 0); del buf213  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf217, permute_303, out=buf218)
    del permute_303
    buf219 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (768, 512), (1, 768), 0), view_154, out=buf219)
    del view_154
    buf220 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf221 = reinterpret_tensor(buf208, (1, 512, 768), (393216, 768, 1), 0); del buf208  # reuse
    buf222 = buf195; del buf195  # reuse
    buf223 = buf194; del buf194  # reuse
    buf224 = buf221; del buf221  # reuse
    buf225 = empty((768, ), device='cpu', dtype=torch.float32)
    buf226 = empty((768, ), device='cpu', dtype=torch.float32)
    buf227 = buf155; del buf155  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43(c_void_p(buf224.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    del div_42
    del getitem_71
    del mul_50
    del primals_116
    buf228 = reinterpret_tensor(buf190, (512, 3072), (3072, 1), 0); del buf190  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (512, 768), (768, 1), 0), permute_307, out=buf228)
    del permute_307
    buf229 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (768, 512), (1, 768), 0), view_152, out=buf229)
    del view_152
    buf230 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf231 = reinterpret_tensor(buf228, (1, 512, 3072), (1572864, 3072, 1), 0); del buf228  # reuse
    cpp_fused_gelu_gelu_backward_sum_44(c_void_p(buf231.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf230.data_ptr()))
    del addmm_40
    buf232 = reinterpret_tensor(buf227, (512, 768), (768, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (512, 3072), (3072, 1), 0), permute_311, out=buf232)
    del permute_311
    buf233 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (3072, 512), (1, 3072), 0), view_150, out=buf233)
    del view_150
    buf234 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf235 = buf223; del buf223  # reuse
    buf236 = buf222; del buf222  # reuse
    buf237 = reinterpret_tensor(buf218, (1, 512, 768), (393216, 768, 1), 0); del buf218  # reuse
    buf238 = empty((768, ), device='cpu', dtype=torch.float32)
    buf239 = empty((768, ), device='cpu', dtype=torch.float32)
    buf240 = reinterpret_tensor(buf217, (1, 512, 768), (393216, 768, 1), 0); del buf217  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_45(c_void_p(buf231.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(mul_45.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    del div_43
    del getitem_67
    del mul_45
    del primals_110
    buf241 = buf232; del buf232  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (512, 768), (768, 1), 0), permute_315, out=buf241)
    del permute_315
    buf242 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (768, 512), (1, 768), 0), view_148, out=buf242)
    del view_148
    buf243 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_46(c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = reinterpret_tensor(buf240, (12, 512, 64), (32768, 64, 1), 0); del buf240  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_31, reinterpret_tensor(buf241, (12, 512, 64), (64, 768, 1), 0), out=buf244)
    del permute_default_31
    buf245 = reinterpret_tensor(buf206, (12, 512, 512), (262144, 512, 1), 0); del buf206  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf241, (12, 512, 64), (64, 768, 1), 0), permute_default_32, out=buf245)
    del permute_default_32
    buf246 = buf205; del buf205  # reuse
    buf247 = reinterpret_tensor(buf245, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf245  # reuse
    cpp_fused_47(c_void_p(buf247.data_ptr()), c_void_p(getitem_137.data_ptr()), c_void_p(alias_default_11.data_ptr()), c_void_p(buf246.data_ptr()))
    del alias_default_11
    del getitem_137
    buf248 = reinterpret_tensor(buf241, (12, 64, 512), (32768, 512, 1), 0); del buf241  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_33, reinterpret_tensor(buf247, (12, 512, 512), (262144, 512, 1), 0), out=buf248)
    del permute_default_33
    buf249 = reinterpret_tensor(buf224, (12, 512, 64), (32768, 64, 1), 0); del buf224  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf247, (12, 512, 512), (262144, 512, 1), 0), permute_default_34, out=buf249)
    del permute_default_34
    buf250 = buf214; del buf214  # reuse
    cpp_fused_view_48(c_void_p(buf244.data_ptr()), c_void_p(buf250.data_ptr()))
    buf251 = reinterpret_tensor(buf244, (512, 768), (768, 1), 0); del buf244  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf250, permute_327, out=buf251)
    del permute_327
    buf252 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf250, (768, 512), (1, 768), 0), view_132, out=buf252)
    buf253 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf254 = reinterpret_tensor(buf248, (512, 768), (1, 512), 0); del buf248  # reuse
    cpp_fused_sum_view_49(c_void_p(buf254.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf253.data_ptr()))
    buf255 = buf250; del buf250  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf254, permute_332, out=buf255)
    del permute_332
    buf256 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (768, 512), (512, 1), 0), view_132, out=buf256)
    buf257 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf258 = buf210; del buf210  # reuse
    cpp_fused_sum_view_50(c_void_p(buf254.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    buf259 = reinterpret_tensor(buf254, (512, 768), (768, 1), 0); del buf254  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf258, permute_336, out=buf259)
    del permute_336
    buf260 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (768, 512), (1, 768), 0), view_132, out=buf260)
    del view_132
    buf261 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf262 = reinterpret_tensor(buf249, (1, 512, 768), (393216, 768, 1), 0); del buf249  # reuse
    buf263 = buf236; del buf236  # reuse
    buf264 = buf235; del buf235  # reuse
    buf265 = buf262; del buf262  # reuse
    buf266 = empty((768, ), device='cpu', dtype=torch.float32)
    buf267 = empty((768, ), device='cpu', dtype=torch.float32)
    buf268 = buf196; del buf196  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51(c_void_p(buf265.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    del div_45
    del getitem_61
    del mul_43
    del primals_100
    buf269 = reinterpret_tensor(buf231, (512, 3072), (3072, 1), 0); del buf231  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf268, (512, 768), (768, 1), 0), permute_340, out=buf269)
    del permute_340
    buf270 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf268, (768, 512), (1, 768), 0), view_130, out=buf270)
    del view_130
    buf271 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf272 = reinterpret_tensor(buf269, (1, 512, 3072), (1572864, 3072, 1), 0); del buf269  # reuse
    cpp_fused_gelu_gelu_backward_sum_52(c_void_p(buf272.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf271.data_ptr()))
    del addmm_34
    buf273 = reinterpret_tensor(buf268, (512, 768), (768, 1), 0); del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (512, 3072), (3072, 1), 0), permute_344, out=buf273)
    del permute_344
    buf274 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (3072, 512), (1, 3072), 0), view_128, out=buf274)
    del view_128
    buf275 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf276 = buf264; del buf264  # reuse
    buf277 = buf263; del buf263  # reuse
    buf278 = reinterpret_tensor(buf259, (1, 512, 768), (393216, 768, 1), 0); del buf259  # reuse
    buf279 = empty((768, ), device='cpu', dtype=torch.float32)
    buf280 = empty((768, ), device='cpu', dtype=torch.float32)
    buf281 = reinterpret_tensor(buf258, (1, 512, 768), (393216, 768, 1), 0); del buf258  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_53(c_void_p(buf272.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(mul_38.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del div_46
    del getitem_57
    del mul_38
    del primals_94
    buf282 = buf273; del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf281, (512, 768), (768, 1), 0), permute_348, out=buf282)
    del permute_348
    buf283 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf281, (768, 512), (1, 768), 0), view_126, out=buf283)
    del view_126
    buf284 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf281.data_ptr()), c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf281, (12, 512, 64), (32768, 64, 1), 0); del buf281  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_37, reinterpret_tensor(buf282, (12, 512, 64), (64, 768, 1), 0), out=buf285)
    del permute_default_37
    buf286 = reinterpret_tensor(buf247, (12, 512, 512), (262144, 512, 1), 0); del buf247  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf282, (12, 512, 64), (64, 768, 1), 0), permute_default_38, out=buf286)
    del permute_default_38
    buf287 = buf246; del buf246  # reuse
    buf288 = reinterpret_tensor(buf286, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf286  # reuse
    cpp_fused_55(c_void_p(buf288.data_ptr()), c_void_p(getitem_139.data_ptr()), c_void_p(alias_default_13.data_ptr()), c_void_p(buf287.data_ptr()))
    del alias_default_13
    del getitem_139
    buf289 = reinterpret_tensor(buf282, (12, 64, 512), (32768, 512, 1), 0); del buf282  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_39, reinterpret_tensor(buf288, (12, 512, 512), (262144, 512, 1), 0), out=buf289)
    del permute_default_39
    buf290 = reinterpret_tensor(buf265, (12, 512, 64), (32768, 64, 1), 0); del buf265  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf288, (12, 512, 512), (262144, 512, 1), 0), permute_default_40, out=buf290)
    del permute_default_40
    buf291 = buf255; del buf255  # reuse
    cpp_fused_view_56(c_void_p(buf285.data_ptr()), c_void_p(buf291.data_ptr()))
    buf292 = reinterpret_tensor(buf285, (512, 768), (768, 1), 0); del buf285  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf291, permute_360, out=buf292)
    del permute_360
    buf293 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf291, (768, 512), (1, 768), 0), view_110, out=buf293)
    buf294 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf295 = reinterpret_tensor(buf289, (512, 768), (1, 512), 0); del buf289  # reuse
    cpp_fused_sum_view_57(c_void_p(buf295.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf294.data_ptr()))
    buf296 = buf291; del buf291  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf295, permute_365, out=buf296)
    del permute_365
    buf297 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (768, 512), (512, 1), 0), view_110, out=buf297)
    buf298 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf299 = buf251; del buf251  # reuse
    cpp_fused_sum_view_58(c_void_p(buf295.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()))
    buf300 = reinterpret_tensor(buf295, (512, 768), (768, 1), 0); del buf295  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf299, permute_369, out=buf300)
    del permute_369
    buf301 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (768, 512), (1, 768), 0), view_110, out=buf301)
    del view_110
    buf302 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf303 = reinterpret_tensor(buf290, (1, 512, 768), (393216, 768, 1), 0); del buf290  # reuse
    buf304 = buf277; del buf277  # reuse
    buf305 = buf276; del buf276  # reuse
    buf306 = buf303; del buf303  # reuse
    buf307 = empty((768, ), device='cpu', dtype=torch.float32)
    buf308 = empty((768, ), device='cpu', dtype=torch.float32)
    buf309 = buf237; del buf237  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_59(c_void_p(buf306.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(mul_36.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()))
    del div_48
    del getitem_51
    del mul_36
    del primals_84
    buf310 = reinterpret_tensor(buf272, (512, 3072), (3072, 1), 0); del buf272  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf309, (512, 768), (768, 1), 0), permute_373, out=buf310)
    del permute_373
    buf311 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf309, (768, 512), (1, 768), 0), view_108, out=buf311)
    del view_108
    buf312 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf313 = reinterpret_tensor(buf310, (1, 512, 3072), (1572864, 3072, 1), 0); del buf310  # reuse
    cpp_fused_gelu_gelu_backward_sum_60(c_void_p(buf313.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf312.data_ptr()))
    del addmm_28
    buf314 = reinterpret_tensor(buf309, (512, 768), (768, 1), 0); del buf309  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf313, (512, 3072), (3072, 1), 0), permute_377, out=buf314)
    del permute_377
    buf315 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf313, (3072, 512), (1, 3072), 0), view_106, out=buf315)
    del view_106
    buf316 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf317 = buf305; del buf305  # reuse
    buf318 = buf304; del buf304  # reuse
    buf319 = reinterpret_tensor(buf300, (1, 512, 768), (393216, 768, 1), 0); del buf300  # reuse
    buf320 = empty((768, ), device='cpu', dtype=torch.float32)
    buf321 = empty((768, ), device='cpu', dtype=torch.float32)
    buf322 = reinterpret_tensor(buf299, (1, 512, 768), (393216, 768, 1), 0); del buf299  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_61(c_void_p(buf313.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(mul_31.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()))
    del div_49
    del getitem_47
    del mul_31
    del primals_78
    buf323 = buf314; del buf314  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (512, 768), (768, 1), 0), permute_381, out=buf323)
    del permute_381
    buf324 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (768, 512), (1, 768), 0), view_104, out=buf324)
    del view_104
    buf325 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_62(c_void_p(buf322.data_ptr()), c_void_p(buf325.data_ptr()))
    buf326 = reinterpret_tensor(buf322, (12, 512, 64), (32768, 64, 1), 0); del buf322  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_43, reinterpret_tensor(buf323, (12, 512, 64), (64, 768, 1), 0), out=buf326)
    del permute_default_43
    buf327 = reinterpret_tensor(buf288, (12, 512, 512), (262144, 512, 1), 0); del buf288  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf323, (12, 512, 64), (64, 768, 1), 0), permute_default_44, out=buf327)
    del permute_default_44
    buf328 = buf287; del buf287  # reuse
    buf329 = reinterpret_tensor(buf327, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf327  # reuse
    cpp_fused_63(c_void_p(buf329.data_ptr()), c_void_p(getitem_141.data_ptr()), c_void_p(alias_default_15.data_ptr()), c_void_p(buf328.data_ptr()))
    del alias_default_15
    del getitem_141
    buf330 = reinterpret_tensor(buf323, (12, 64, 512), (32768, 512, 1), 0); del buf323  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_45, reinterpret_tensor(buf329, (12, 512, 512), (262144, 512, 1), 0), out=buf330)
    del permute_default_45
    buf331 = reinterpret_tensor(buf306, (12, 512, 64), (32768, 64, 1), 0); del buf306  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf329, (12, 512, 512), (262144, 512, 1), 0), permute_default_46, out=buf331)
    del permute_default_46
    buf332 = buf296; del buf296  # reuse
    cpp_fused_view_64(c_void_p(buf326.data_ptr()), c_void_p(buf332.data_ptr()))
    buf333 = reinterpret_tensor(buf326, (512, 768), (768, 1), 0); del buf326  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf332, permute_393, out=buf333)
    del permute_393
    buf334 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf332, (768, 512), (1, 768), 0), view_88, out=buf334)
    buf335 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf336 = reinterpret_tensor(buf330, (512, 768), (1, 512), 0); del buf330  # reuse
    cpp_fused_sum_view_65(c_void_p(buf336.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf335.data_ptr()))
    buf337 = buf332; del buf332  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf336, permute_398, out=buf337)
    del permute_398
    buf338 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (768, 512), (512, 1), 0), view_88, out=buf338)
    buf339 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf340 = buf292; del buf292  # reuse
    cpp_fused_sum_view_66(c_void_p(buf336.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    buf341 = reinterpret_tensor(buf336, (512, 768), (768, 1), 0); del buf336  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf340, permute_402, out=buf341)
    del permute_402
    buf342 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf340, (768, 512), (1, 768), 0), view_88, out=buf342)
    del view_88
    buf343 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf344 = reinterpret_tensor(buf331, (1, 512, 768), (393216, 768, 1), 0); del buf331  # reuse
    buf345 = buf318; del buf318  # reuse
    buf346 = buf317; del buf317  # reuse
    buf347 = buf344; del buf344  # reuse
    buf348 = empty((768, ), device='cpu', dtype=torch.float32)
    buf349 = empty((768, ), device='cpu', dtype=torch.float32)
    buf350 = buf278; del buf278  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67(c_void_p(buf347.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(mul_29.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    del div_51
    del getitem_41
    del mul_29
    del primals_68
    buf351 = reinterpret_tensor(buf313, (512, 3072), (3072, 1), 0); del buf313  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (512, 768), (768, 1), 0), permute_406, out=buf351)
    del permute_406
    buf352 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (768, 512), (1, 768), 0), view_86, out=buf352)
    del view_86
    buf353 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf354 = reinterpret_tensor(buf351, (1, 512, 3072), (1572864, 3072, 1), 0); del buf351  # reuse
    cpp_fused_gelu_gelu_backward_sum_68(c_void_p(buf354.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf353.data_ptr()))
    del addmm_22
    buf355 = reinterpret_tensor(buf350, (512, 768), (768, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf354, (512, 3072), (3072, 1), 0), permute_410, out=buf355)
    del permute_410
    buf356 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf354, (3072, 512), (1, 3072), 0), view_84, out=buf356)
    del view_84
    buf357 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf358 = buf346; del buf346  # reuse
    buf359 = buf345; del buf345  # reuse
    buf360 = reinterpret_tensor(buf341, (1, 512, 768), (393216, 768, 1), 0); del buf341  # reuse
    buf361 = empty((768, ), device='cpu', dtype=torch.float32)
    buf362 = empty((768, ), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf340, (1, 512, 768), (393216, 768, 1), 0); del buf340  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_69(c_void_p(buf354.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    del div_52
    del getitem_37
    del mul_24
    del primals_62
    buf364 = buf355; del buf355  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (512, 768), (768, 1), 0), permute_414, out=buf364)
    del permute_414
    buf365 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (768, 512), (1, 768), 0), view_82, out=buf365)
    del view_82
    buf366 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_70(c_void_p(buf363.data_ptr()), c_void_p(buf366.data_ptr()))
    buf367 = reinterpret_tensor(buf363, (12, 512, 64), (32768, 64, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_49, reinterpret_tensor(buf364, (12, 512, 64), (64, 768, 1), 0), out=buf367)
    del permute_default_49
    buf368 = reinterpret_tensor(buf329, (12, 512, 512), (262144, 512, 1), 0); del buf329  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf364, (12, 512, 64), (64, 768, 1), 0), permute_default_50, out=buf368)
    del permute_default_50
    buf369 = buf328; del buf328  # reuse
    buf370 = reinterpret_tensor(buf368, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf368  # reuse
    cpp_fused_71(c_void_p(buf370.data_ptr()), c_void_p(getitem_143.data_ptr()), c_void_p(alias_default_17.data_ptr()), c_void_p(buf369.data_ptr()))
    del alias_default_17
    del getitem_143
    buf371 = reinterpret_tensor(buf364, (12, 64, 512), (32768, 512, 1), 0); del buf364  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_51, reinterpret_tensor(buf370, (12, 512, 512), (262144, 512, 1), 0), out=buf371)
    del permute_default_51
    buf372 = reinterpret_tensor(buf347, (12, 512, 64), (32768, 64, 1), 0); del buf347  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf370, (12, 512, 512), (262144, 512, 1), 0), permute_default_52, out=buf372)
    del permute_default_52
    buf373 = buf337; del buf337  # reuse
    cpp_fused_view_72(c_void_p(buf367.data_ptr()), c_void_p(buf373.data_ptr()))
    buf374 = reinterpret_tensor(buf367, (512, 768), (768, 1), 0); del buf367  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf373, permute_426, out=buf374)
    del permute_426
    buf375 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf373, (768, 512), (1, 768), 0), view_66, out=buf375)
    buf376 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf377 = reinterpret_tensor(buf371, (512, 768), (1, 512), 0); del buf371  # reuse
    cpp_fused_sum_view_73(c_void_p(buf377.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf376.data_ptr()))
    buf378 = buf373; del buf373  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf377, permute_431, out=buf378)
    del permute_431
    buf379 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (768, 512), (512, 1), 0), view_66, out=buf379)
    buf380 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf381 = buf333; del buf333  # reuse
    cpp_fused_sum_view_74(c_void_p(buf377.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = reinterpret_tensor(buf377, (512, 768), (768, 1), 0); del buf377  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf381, permute_435, out=buf382)
    del permute_435
    buf383 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (768, 512), (1, 768), 0), view_66, out=buf383)
    del view_66
    buf384 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf385 = reinterpret_tensor(buf372, (1, 512, 768), (393216, 768, 1), 0); del buf372  # reuse
    buf386 = buf359; del buf359  # reuse
    buf387 = buf358; del buf358  # reuse
    buf388 = buf385; del buf385  # reuse
    buf389 = empty((768, ), device='cpu', dtype=torch.float32)
    buf390 = empty((768, ), device='cpu', dtype=torch.float32)
    buf391 = buf319; del buf319  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_75(c_void_p(buf388.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(mul_22.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    del div_54
    del getitem_31
    del mul_22
    del primals_52
    buf392 = reinterpret_tensor(buf354, (512, 3072), (3072, 1), 0); del buf354  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (512, 768), (768, 1), 0), permute_439, out=buf392)
    del permute_439
    buf393 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (768, 512), (1, 768), 0), view_64, out=buf393)
    del view_64
    buf394 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf395 = reinterpret_tensor(buf392, (1, 512, 3072), (1572864, 3072, 1), 0); del buf392  # reuse
    cpp_fused_gelu_gelu_backward_sum_76(c_void_p(buf395.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf394.data_ptr()))
    del addmm_16
    buf396 = reinterpret_tensor(buf391, (512, 768), (768, 1), 0); del buf391  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (512, 3072), (3072, 1), 0), permute_443, out=buf396)
    del permute_443
    buf397 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (3072, 512), (1, 3072), 0), view_62, out=buf397)
    del view_62
    buf398 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf399 = buf387; del buf387  # reuse
    buf400 = buf386; del buf386  # reuse
    buf401 = reinterpret_tensor(buf382, (1, 512, 768), (393216, 768, 1), 0); del buf382  # reuse
    buf402 = empty((768, ), device='cpu', dtype=torch.float32)
    buf403 = empty((768, ), device='cpu', dtype=torch.float32)
    buf404 = reinterpret_tensor(buf381, (1, 512, 768), (393216, 768, 1), 0); del buf381  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_77(c_void_p(buf395.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del div_55
    del getitem_27
    del mul_17
    del primals_46
    buf405 = buf396; del buf396  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf404, (512, 768), (768, 1), 0), permute_447, out=buf405)
    del permute_447
    buf406 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf404, (768, 512), (1, 768), 0), view_60, out=buf406)
    del view_60
    buf407 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_78(c_void_p(buf404.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = reinterpret_tensor(buf404, (12, 512, 64), (32768, 64, 1), 0); del buf404  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_55, reinterpret_tensor(buf405, (12, 512, 64), (64, 768, 1), 0), out=buf408)
    del permute_default_55
    buf409 = reinterpret_tensor(buf370, (12, 512, 512), (262144, 512, 1), 0); del buf370  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf405, (12, 512, 64), (64, 768, 1), 0), permute_default_56, out=buf409)
    del permute_default_56
    buf410 = buf369; del buf369  # reuse
    buf411 = reinterpret_tensor(buf409, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf409  # reuse
    cpp_fused_79(c_void_p(buf411.data_ptr()), c_void_p(getitem_145.data_ptr()), c_void_p(alias_default_19.data_ptr()), c_void_p(buf410.data_ptr()))
    del alias_default_19
    del getitem_145
    buf412 = reinterpret_tensor(buf405, (12, 64, 512), (32768, 512, 1), 0); del buf405  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_57, reinterpret_tensor(buf411, (12, 512, 512), (262144, 512, 1), 0), out=buf412)
    del permute_default_57
    buf413 = reinterpret_tensor(buf388, (12, 512, 64), (32768, 64, 1), 0); del buf388  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf411, (12, 512, 512), (262144, 512, 1), 0), permute_default_58, out=buf413)
    del permute_default_58
    buf414 = buf378; del buf378  # reuse
    cpp_fused_view_80(c_void_p(buf408.data_ptr()), c_void_p(buf414.data_ptr()))
    buf415 = reinterpret_tensor(buf408, (512, 768), (768, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf414, permute_459, out=buf415)
    del permute_459
    buf416 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf414, (768, 512), (1, 768), 0), view_44, out=buf416)
    buf417 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf418 = reinterpret_tensor(buf412, (512, 768), (1, 512), 0); del buf412  # reuse
    cpp_fused_sum_view_81(c_void_p(buf418.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf417.data_ptr()))
    buf419 = buf414; del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf418, permute_464, out=buf419)
    del permute_464
    buf420 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf418, (768, 512), (512, 1), 0), view_44, out=buf420)
    buf421 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf422 = buf374; del buf374  # reuse
    cpp_fused_sum_view_82(c_void_p(buf418.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()))
    buf423 = reinterpret_tensor(buf418, (512, 768), (768, 1), 0); del buf418  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf422, permute_468, out=buf423)
    del permute_468
    buf424 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (768, 512), (1, 768), 0), view_44, out=buf424)
    del view_44
    buf425 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf426 = reinterpret_tensor(buf413, (1, 512, 768), (393216, 768, 1), 0); del buf413  # reuse
    buf427 = buf400; del buf400  # reuse
    buf428 = buf399; del buf399  # reuse
    buf429 = buf426; del buf426  # reuse
    buf430 = empty((768, ), device='cpu', dtype=torch.float32)
    buf431 = empty((768, ), device='cpu', dtype=torch.float32)
    buf432 = buf360; del buf360  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83(c_void_p(buf429.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(mul_15.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()))
    del div_57
    del getitem_21
    del mul_15
    del primals_36
    buf433 = reinterpret_tensor(buf395, (512, 3072), (3072, 1), 0); del buf395  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (512, 768), (768, 1), 0), permute_472, out=buf433)
    del permute_472
    buf434 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (768, 512), (1, 768), 0), view_42, out=buf434)
    del view_42
    buf435 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf436 = reinterpret_tensor(buf433, (1, 512, 3072), (1572864, 3072, 1), 0); del buf433  # reuse
    cpp_fused_gelu_gelu_backward_sum_84(c_void_p(buf436.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf435.data_ptr()))
    del addmm_10
    buf437 = reinterpret_tensor(buf432, (512, 768), (768, 1), 0); del buf432  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (512, 3072), (3072, 1), 0), permute_476, out=buf437)
    del permute_476
    buf438 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (3072, 512), (1, 3072), 0), view_40, out=buf438)
    del view_40
    buf439 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf440 = buf428; del buf428  # reuse
    buf441 = buf427; del buf427  # reuse
    buf442 = reinterpret_tensor(buf423, (1, 512, 768), (393216, 768, 1), 0); del buf423  # reuse
    buf443 = empty((768, ), device='cpu', dtype=torch.float32)
    buf444 = empty((768, ), device='cpu', dtype=torch.float32)
    buf445 = reinterpret_tensor(buf422, (1, 512, 768), (393216, 768, 1), 0); del buf422  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_85(c_void_p(buf436.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()))
    del div_58
    del getitem_17
    del mul_10
    del primals_30
    buf446 = buf437; del buf437  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf445, (512, 768), (768, 1), 0), permute_480, out=buf446)
    del permute_480
    buf447 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf445, (768, 512), (1, 768), 0), view_38, out=buf447)
    del view_38
    buf448 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_86(c_void_p(buf445.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = reinterpret_tensor(buf445, (12, 512, 64), (32768, 64, 1), 0); del buf445  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_61, reinterpret_tensor(buf446, (12, 512, 64), (64, 768, 1), 0), out=buf449)
    del permute_default_61
    buf450 = reinterpret_tensor(buf411, (12, 512, 512), (262144, 512, 1), 0); del buf411  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf446, (12, 512, 64), (64, 768, 1), 0), permute_default_62, out=buf450)
    del permute_default_62
    buf451 = buf410; del buf410  # reuse
    buf452 = reinterpret_tensor(buf450, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf450  # reuse
    cpp_fused_87(c_void_p(buf452.data_ptr()), c_void_p(getitem_147.data_ptr()), c_void_p(alias_default_21.data_ptr()), c_void_p(buf451.data_ptr()))
    del alias_default_21
    del getitem_147
    buf453 = reinterpret_tensor(buf446, (12, 64, 512), (32768, 512, 1), 0); del buf446  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_63, reinterpret_tensor(buf452, (12, 512, 512), (262144, 512, 1), 0), out=buf453)
    del permute_default_63
    buf454 = reinterpret_tensor(buf429, (12, 512, 64), (32768, 64, 1), 0); del buf429  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf452, (12, 512, 512), (262144, 512, 1), 0), permute_default_64, out=buf454)
    del permute_default_64
    buf455 = buf419; del buf419  # reuse
    cpp_fused_view_88(c_void_p(buf449.data_ptr()), c_void_p(buf455.data_ptr()))
    buf456 = reinterpret_tensor(buf449, (512, 768), (768, 1), 0); del buf449  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf455, permute_492, out=buf456)
    del permute_492
    buf457 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf455, (768, 512), (1, 768), 0), view_22, out=buf457)
    buf458 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf459 = reinterpret_tensor(buf453, (512, 768), (1, 512), 0); del buf453  # reuse
    cpp_fused_sum_view_89(c_void_p(buf459.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf458.data_ptr()))
    buf460 = buf455; del buf455  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf459, permute_497, out=buf460)
    del permute_497
    buf461 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (768, 512), (512, 1), 0), view_22, out=buf461)
    buf462 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf463 = buf415; del buf415  # reuse
    cpp_fused_sum_view_90(c_void_p(buf459.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    buf464 = reinterpret_tensor(buf459, (512, 768), (768, 1), 0); del buf459  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf463, permute_501, out=buf464)
    del permute_501
    buf465 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf463, (768, 512), (1, 768), 0), view_22, out=buf465)
    del view_22
    buf466 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf467 = reinterpret_tensor(buf454, (1, 512, 768), (393216, 768, 1), 0); del buf454  # reuse
    buf468 = buf441; del buf441  # reuse
    buf469 = buf440; del buf440  # reuse
    buf470 = buf467; del buf467  # reuse
    buf471 = empty((768, ), device='cpu', dtype=torch.float32)
    buf472 = empty((768, ), device='cpu', dtype=torch.float32)
    buf473 = buf401; del buf401  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_91(c_void_p(buf470.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()))
    del div_60
    del getitem_11
    del mul_8
    del primals_20
    buf474 = reinterpret_tensor(buf436, (512, 3072), (3072, 1), 0); del buf436  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf473, (512, 768), (768, 1), 0), permute_505, out=buf474)
    del permute_505
    buf475 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf473, (768, 512), (1, 768), 0), view_20, out=buf475)
    del view_20
    buf476 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf477 = reinterpret_tensor(buf474, (1, 512, 3072), (1572864, 3072, 1), 0); del buf474  # reuse
    cpp_fused_gelu_gelu_backward_sum_92(c_void_p(buf477.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf476.data_ptr()))
    del addmm_4
    buf478 = reinterpret_tensor(buf473, (512, 768), (768, 1), 0); del buf473  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf477, (512, 3072), (3072, 1), 0), permute_509, out=buf478)
    del permute_509
    buf479 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf477, (3072, 512), (1, 3072), 0), view_18, out=buf479)
    del view_18
    buf480 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf481 = buf469; del buf469  # reuse
    buf482 = buf468; del buf468  # reuse
    buf483 = reinterpret_tensor(buf464, (1, 512, 768), (393216, 768, 1), 0); del buf464  # reuse
    buf484 = empty((768, ), device='cpu', dtype=torch.float32)
    buf485 = empty((768, ), device='cpu', dtype=torch.float32)
    buf486 = reinterpret_tensor(buf463, (1, 512, 768), (393216, 768, 1), 0); del buf463  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_93(c_void_p(buf477.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    del buf477
    del div_61
    del getitem_7
    del mul_3
    del primals_14
    buf487 = buf478; del buf478  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (512, 768), (768, 1), 0), permute_513, out=buf487)
    del permute_513
    buf488 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (768, 512), (1, 768), 0), view_16, out=buf488)
    del view_16
    buf489 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_94(c_void_p(buf486.data_ptr()), c_void_p(buf489.data_ptr()))
    buf490 = reinterpret_tensor(buf486, (12, 512, 64), (32768, 64, 1), 0); del buf486  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_67, reinterpret_tensor(buf487, (12, 512, 64), (64, 768, 1), 0), out=buf490)
    del permute_default_67
    buf491 = reinterpret_tensor(buf452, (12, 512, 512), (262144, 512, 1), 0); del buf452  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf487, (12, 512, 64), (64, 768, 1), 0), permute_default_68, out=buf491)
    del permute_default_68
    buf492 = buf451; del buf451  # reuse
    buf493 = reinterpret_tensor(buf491, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf491  # reuse
    cpp_fused_95(c_void_p(buf493.data_ptr()), c_void_p(getitem_149.data_ptr()), c_void_p(alias_default_23.data_ptr()), c_void_p(buf492.data_ptr()))
    del alias_default_23
    del buf492
    del getitem_149
    buf494 = reinterpret_tensor(buf487, (12, 64, 512), (32768, 512, 1), 0); del buf487  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_69, reinterpret_tensor(buf493, (12, 512, 512), (262144, 512, 1), 0), out=buf494)
    del permute_default_69
    buf495 = reinterpret_tensor(buf470, (12, 512, 64), (32768, 64, 1), 0); del buf470  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf493, (12, 512, 512), (262144, 512, 1), 0), permute_default_70, out=buf495)
    del buf493
    del permute_default_70
    buf496 = buf460; del buf460  # reuse
    cpp_fused_view_96(c_void_p(buf490.data_ptr()), c_void_p(buf496.data_ptr()))
    buf497 = reinterpret_tensor(buf490, (512, 768), (768, 1), 0); del buf490  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf496, permute_525, out=buf497)
    del permute_525
    buf498 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf496, (768, 512), (1, 768), 0), view, out=buf498)
    buf499 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf500 = reinterpret_tensor(buf494, (512, 768), (1, 512), 0); del buf494  # reuse
    cpp_fused_sum_view_97(c_void_p(buf500.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf499.data_ptr()))
    buf501 = buf496; del buf496  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf500, permute_530, out=buf501)
    del permute_530
    buf502 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf500, (768, 512), (512, 1), 0), view, out=buf502)
    buf503 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf504 = buf456; del buf456  # reuse
    cpp_fused_sum_view_98(c_void_p(buf500.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()))
    buf505 = reinterpret_tensor(buf500, (512, 768), (768, 1), 0); del buf500  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf504, permute_534, out=buf505)
    del permute_534
    buf506 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf504, (768, 512), (1, 768), 0), view, out=buf506)
    del view
    buf507 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf508 = buf483; del buf483  # reuse
    buf509 = buf482; del buf482  # reuse
    buf510 = buf481; del buf481  # reuse
    buf515 = reinterpret_tensor(buf495, (1, 512, 768), (393216, 768, 1), 0); del buf495  # reuse
    buf519 = buf442; del buf442  # reuse
    buf523 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf512 = empty((768, ), device='cpu', dtype=torch.float32)
    buf513 = empty((768, ), device='cpu', dtype=torch.float32)
    buf514 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_99(c_void_p(buf508.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_63.data_ptr()), c_void_p(slice_4.data_ptr()), c_void_p(expand.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf514.data_ptr()))
    del buf497
    del buf501
    del buf504
    del buf505
    del buf508
    del buf509
    del buf510
    del div_63
    del getitem_3
    del mul_1
    del primals_4
    aten.index_put_(buf514, [slice_4], buf515, True)
    del buf515
    del slice_4
    buf518 = empty((2, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_100(c_void_p(buf518.data_ptr()))
    aten.index_put_(buf518, [expand], buf519, True)
    del buf519
    del expand
    buf522 = empty((30522, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_101(c_void_p(buf522.data_ptr()))
    aten.index_put_(buf522, [primals_206], buf523, True)
    del buf523
    del primals_206
    return (buf522, buf518, buf514, buf512, buf513, reinterpret_tensor(buf506, (768, 768), (768, 1), 0), reinterpret_tensor(buf507, (768, ), (1, ), 0), reinterpret_tensor(buf502, (768, 768), (768, 1), 0), reinterpret_tensor(buf503, (768, ), (1, ), 0), reinterpret_tensor(buf498, (768, 768), (768, 1), 0), reinterpret_tensor(buf499, (768, ), (1, ), 0), reinterpret_tensor(buf488, (768, 768), (768, 1), 0), reinterpret_tensor(buf489, (768, ), (1, ), 0), buf484, buf485, reinterpret_tensor(buf479, (3072, 768), (768, 1), 0), reinterpret_tensor(buf480, (3072, ), (1, ), 0), reinterpret_tensor(buf475, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf476, (768, ), (1, ), 0), buf471, buf472, reinterpret_tensor(buf465, (768, 768), (768, 1), 0), reinterpret_tensor(buf466, (768, ), (1, ), 0), reinterpret_tensor(buf461, (768, 768), (768, 1), 0), reinterpret_tensor(buf462, (768, ), (1, ), 0), reinterpret_tensor(buf457, (768, 768), (768, 1), 0), reinterpret_tensor(buf458, (768, ), (1, ), 0), reinterpret_tensor(buf447, (768, 768), (768, 1), 0), reinterpret_tensor(buf448, (768, ), (1, ), 0), buf443, buf444, reinterpret_tensor(buf438, (3072, 768), (768, 1), 0), reinterpret_tensor(buf439, (3072, ), (1, ), 0), reinterpret_tensor(buf434, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf435, (768, ), (1, ), 0), buf430, buf431, reinterpret_tensor(buf424, (768, 768), (768, 1), 0), reinterpret_tensor(buf425, (768, ), (1, ), 0), reinterpret_tensor(buf420, (768, 768), (768, 1), 0), reinterpret_tensor(buf421, (768, ), (1, ), 0), reinterpret_tensor(buf416, (768, 768), (768, 1), 0), reinterpret_tensor(buf417, (768, ), (1, ), 0), reinterpret_tensor(buf406, (768, 768), (768, 1), 0), reinterpret_tensor(buf407, (768, ), (1, ), 0), buf402, buf403, reinterpret_tensor(buf397, (3072, 768), (768, 1), 0), reinterpret_tensor(buf398, (3072, ), (1, ), 0), reinterpret_tensor(buf393, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf394, (768, ), (1, ), 0), buf389, buf390, reinterpret_tensor(buf383, (768, 768), (768, 1), 0), reinterpret_tensor(buf384, (768, ), (1, ), 0), reinterpret_tensor(buf379, (768, 768), (768, 1), 0), reinterpret_tensor(buf380, (768, ), (1, ), 0), reinterpret_tensor(buf375, (768, 768), (768, 1), 0), reinterpret_tensor(buf376, (768, ), (1, ), 0), reinterpret_tensor(buf365, (768, 768), (768, 1), 0), reinterpret_tensor(buf366, (768, ), (1, ), 0), buf361, buf362, reinterpret_tensor(buf356, (3072, 768), (768, 1), 0), reinterpret_tensor(buf357, (3072, ), (1, ), 0), reinterpret_tensor(buf352, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf353, (768, ), (1, ), 0), buf348, buf349, reinterpret_tensor(buf342, (768, 768), (768, 1), 0), reinterpret_tensor(buf343, (768, ), (1, ), 0), reinterpret_tensor(buf338, (768, 768), (768, 1), 0), reinterpret_tensor(buf339, (768, ), (1, ), 0), reinterpret_tensor(buf334, (768, 768), (768, 1), 0), reinterpret_tensor(buf335, (768, ), (1, ), 0), reinterpret_tensor(buf324, (768, 768), (768, 1), 0), reinterpret_tensor(buf325, (768, ), (1, ), 0), buf320, buf321, reinterpret_tensor(buf315, (3072, 768), (768, 1), 0), reinterpret_tensor(buf316, (3072, ), (1, ), 0), reinterpret_tensor(buf311, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf312, (768, ), (1, ), 0), buf307, buf308, reinterpret_tensor(buf301, (768, 768), (768, 1), 0), reinterpret_tensor(buf302, (768, ), (1, ), 0), reinterpret_tensor(buf297, (768, 768), (768, 1), 0), reinterpret_tensor(buf298, (768, ), (1, ), 0), reinterpret_tensor(buf293, (768, 768), (768, 1), 0), reinterpret_tensor(buf294, (768, ), (1, ), 0), reinterpret_tensor(buf283, (768, 768), (768, 1), 0), reinterpret_tensor(buf284, (768, ), (1, ), 0), buf279, buf280, reinterpret_tensor(buf274, (3072, 768), (768, 1), 0), reinterpret_tensor(buf275, (3072, ), (1, ), 0), reinterpret_tensor(buf270, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf271, (768, ), (1, ), 0), buf266, buf267, reinterpret_tensor(buf260, (768, 768), (768, 1), 0), reinterpret_tensor(buf261, (768, ), (1, ), 0), reinterpret_tensor(buf256, (768, 768), (768, 1), 0), reinterpret_tensor(buf257, (768, ), (1, ), 0), reinterpret_tensor(buf252, (768, 768), (768, 1), 0), reinterpret_tensor(buf253, (768, ), (1, ), 0), reinterpret_tensor(buf242, (768, 768), (768, 1), 0), reinterpret_tensor(buf243, (768, ), (1, ), 0), buf238, buf239, reinterpret_tensor(buf233, (3072, 768), (768, 1), 0), reinterpret_tensor(buf234, (3072, ), (1, ), 0), reinterpret_tensor(buf229, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf230, (768, ), (1, ), 0), buf225, buf226, reinterpret_tensor(buf219, (768, 768), (768, 1), 0), reinterpret_tensor(buf220, (768, ), (1, ), 0), reinterpret_tensor(buf215, (768, 768), (768, 1), 0), reinterpret_tensor(buf216, (768, ), (1, ), 0), reinterpret_tensor(buf211, (768, 768), (768, 1), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), reinterpret_tensor(buf201, (768, 768), (768, 1), 0), reinterpret_tensor(buf202, (768, ), (1, ), 0), buf197, buf198, reinterpret_tensor(buf192, (3072, 768), (768, 1), 0), reinterpret_tensor(buf193, (3072, ), (1, ), 0), reinterpret_tensor(buf188, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf189, (768, ), (1, ), 0), buf184, buf185, reinterpret_tensor(buf178, (768, 768), (768, 1), 0), reinterpret_tensor(buf179, (768, ), (1, ), 0), reinterpret_tensor(buf174, (768, 768), (768, 1), 0), reinterpret_tensor(buf175, (768, ), (1, ), 0), reinterpret_tensor(buf170, (768, 768), (768, 1), 0), reinterpret_tensor(buf171, (768, ), (1, ), 0), reinterpret_tensor(buf160, (768, 768), (768, 1), 0), reinterpret_tensor(buf161, (768, ), (1, ), 0), buf156, buf157, reinterpret_tensor(buf151, (3072, 768), (768, 1), 0), reinterpret_tensor(buf152, (3072, ), (1, ), 0), reinterpret_tensor(buf147, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf148, (768, ), (1, ), 0), buf143, buf144, reinterpret_tensor(buf137, (768, 768), (768, 1), 0), reinterpret_tensor(buf138, (768, ), (1, ), 0), reinterpret_tensor(buf133, (768, 768), (768, 1), 0), reinterpret_tensor(buf134, (768, ), (1, ), 0), reinterpret_tensor(buf129, (768, 768), (768, 1), 0), reinterpret_tensor(buf130, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, 768), (768, 1), 0), reinterpret_tensor(buf120, (768, ), (1, ), 0), buf115, buf116, reinterpret_tensor(buf110, (3072, 768), (768, 1), 0), reinterpret_tensor(buf111, (3072, ), (1, ), 0), reinterpret_tensor(buf106, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf107, (768, ), (1, ), 0), buf102, buf103, reinterpret_tensor(buf96, (768, 768), (768, 1), 0), reinterpret_tensor(buf97, (768, ), (1, ), 0), reinterpret_tensor(buf92, (768, 768), (768, 1), 0), reinterpret_tensor(buf93, (768, ), (1, ), 0), reinterpret_tensor(buf88, (768, 768), (768, 1), 0), reinterpret_tensor(buf89, (768, ), (1, ), 0), reinterpret_tensor(buf78, (768, 768), (768, 1), 0), reinterpret_tensor(buf79, (768, ), (1, ), 0), buf74, buf75, reinterpret_tensor(buf69, (3072, 768), (768, 1), 0), reinterpret_tensor(buf70, (3072, ), (1, ), 0), reinterpret_tensor(buf65, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf66, (768, ), (1, ), 0), buf61, buf62, reinterpret_tensor(buf55, (768, 768), (768, 1), 0), reinterpret_tensor(buf56, (768, ), (1, ), 0), reinterpret_tensor(buf51, (768, 768), (768, 1), 0), reinterpret_tensor(buf52, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, 768), (768, 1), 0), reinterpret_tensor(buf48, (768, ), (1, ), 0), reinterpret_tensor(buf37, (768, 768), (768, 1), 0), reinterpret_tensor(buf38, (768, ), (1, ), 0), buf33, buf34, reinterpret_tensor(buf28, (3072, 768), (768, 1), 0), reinterpret_tensor(buf29, (3072, ), (1, ), 0), reinterpret_tensor(buf24, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf25, (768, ), (1, ), 0), buf20, buf21, reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf16, (768, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf7, (30522, 768), (768, 1), 0), reinterpret_tensor(buf8, (30522, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_207 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_4 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_149 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_67 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_68 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_23 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_69 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_70 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_61 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_62 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_21 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_63 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_64 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_145 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_55 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_56 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_19 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_57 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_58 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_60 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_143 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_49 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_50 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_17 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_51 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_52 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_82 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_141 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_43 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_44 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_15 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_45 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_46 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_139 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_37 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_38 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_13 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_39 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_40 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_137 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_31 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_32 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_11 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_33 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_34 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_148 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_25 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_26 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_9 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_27 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_28 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_133 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_19 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_20 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_7 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_21 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_22 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_192 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_13 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_14 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_5 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_15 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_16 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_129 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_7 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_8 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_3 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_9 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_10 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_236 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_1 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_2 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_1 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_3 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_4 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_258 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_260 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_264 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_72 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_90 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_266 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_40 = rand_strided((512, 30522), (30522, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_134 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_365 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_369 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_402 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_447 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_480 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_497 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_513 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_530 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_534 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 30522), (15627264, 30522, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, primals_207, expand, slice_4, mul_1, getitem_3, view, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, addmm_72, mul_90, view_266, sub_40, convert_element_type, permute_134, div_26, permute_138, div_27, permute_142, permute_146, div_28, permute_150, permute_162, permute_167, permute_171, div_30, permute_175, permute_179, div_31, permute_183, permute_195, permute_200, permute_204, div_33, permute_208, permute_212, div_34, permute_216, permute_228, permute_233, permute_237, div_36, permute_241, permute_245, div_37, permute_249, permute_261, permute_266, permute_270, div_39, permute_274, permute_278, div_40, permute_282, permute_294, permute_299, permute_303, div_42, permute_307, permute_311, div_43, permute_315, permute_327, permute_332, permute_336, div_45, permute_340, permute_344, div_46, permute_348, permute_360, permute_365, permute_369, div_48, permute_373, permute_377, div_49, permute_381, permute_393, permute_398, permute_402, div_51, permute_406, permute_410, div_52, permute_414, permute_426, permute_431, permute_435, div_54, permute_439, permute_443, div_55, permute_447, permute_459, permute_464, permute_468, div_57, permute_472, permute_476, div_58, permute_480, permute_492, permute_497, permute_501, div_60, permute_505, permute_509, div_61, permute_513, permute_525, permute_530, permute_534, div_63, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BertForMaskedLM', benchmark_compiled_module)
