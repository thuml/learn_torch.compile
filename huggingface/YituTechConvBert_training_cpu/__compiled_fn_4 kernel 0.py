
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_13 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_17 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_18 = async_compile.cpp('''
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


cpp_fused_sum_19 = async_compile.cpp('''
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


cpp_fused_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_21 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_26 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_29 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_30 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_31 = async_compile.cpp('''
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


cpp_fused_sum_32 = async_compile.cpp('''
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


cpp_fused_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_39 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_42 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_43 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_44 = async_compile.cpp('''
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


cpp_fused_sum_45 = async_compile.cpp('''
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


cpp_fused_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_52 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_55 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_56 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57 = async_compile.cpp('''
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


cpp_fused_sum_58 = async_compile.cpp('''
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


cpp_fused_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_60 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_68 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_69 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_70 = async_compile.cpp('''
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


cpp_fused_sum_71 = async_compile.cpp('''
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


cpp_fused_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_73 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_78 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_81 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_82 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83 = async_compile.cpp('''
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


cpp_fused_sum_84 = async_compile.cpp('''
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


cpp_fused_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_86 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_91 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_94 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_95 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_96 = async_compile.cpp('''
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


cpp_fused_sum_97 = async_compile.cpp('''
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


cpp_fused_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_99 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_104 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_107 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_108 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_109 = async_compile.cpp('''
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


cpp_fused_sum_110 = async_compile.cpp('''
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


cpp_fused_111 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_117 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_120 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_121 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_122 = async_compile.cpp('''
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


cpp_fused_sum_123 = async_compile.cpp('''
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


cpp_fused_124 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_125 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_130 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_133 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_134 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_135 = async_compile.cpp('''
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


cpp_fused_sum_136 = async_compile.cpp('''
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


cpp_fused_137 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_138 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_143 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_146 = async_compile.cpp('''
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
                       const bool* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp2, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp10, 8);
                        float tmp13[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp13, 8);
                        float tmp16[8*8] __attribute__ ((aligned (8)));
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr7 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp20, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1) + (512L*x1_inner)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(tmp10 + static_cast<long>(8L*x1_inner));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x1_inner));
                            auto tmp17 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = tmp1 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp12 = tmp9 + tmp11;
                            auto tmp15 = tmp12 + tmp14;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp15 * tmp18;
                            auto tmp22 = tmp19 * tmp21;
                            tmp15.store(tmp16 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp22;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp16, 8, in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr8[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
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
                auto tmp1 = in_ptr9[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_147 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_148 = async_compile.cpp('''
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


cpp_fused_sum_149 = async_compile.cpp('''
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


cpp_fused_150 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                        }
                        out_ptr0[static_cast<long>(x1 + (6L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (6L*x2) + (3072L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (6L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x0) + (262144L*x1))] = tmp10;
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_col2im_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(199680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(4L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(520);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr0[static_cast<long>(4L + x1 + (520L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp8;
                        tmp_acc0 = tmp_acc0 + tmp8;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    tmp_acc0 = tmp_acc0 + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (54L*x1)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer((x0 + x0_inner), 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp2 - tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(54L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (54L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (54L*x1))];
                    auto tmp3 = out_ptr0[static_cast<long>((6L*x1) + (c10::div_floor_integer(x0, 9L)))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (9L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (9L*x0))];
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_transpose_view_155 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 * tmp5;
                    auto tmp8 = static_cast<float>(0.3535533905932738);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp6 + tmp10;
                    tmp2.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x0 + (512L*x1) + (512L*x1_inner))] = tmpbuf[x1_inner]; }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1) + (384L*x1_inner)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x0)), static_cast<long>(512L));
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_156 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196608L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const long* in_ptr10,
                       const long* in_ptr11,
                       const long* in_ptr12,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
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
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (768L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (512L*x1))];
                        auto tmp5 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x1 + (768L*x0))];
                        auto tmp9 = in_ptr5[static_cast<long>(x1 + (768L*x0))];
                        auto tmp11 = in_ptr6[static_cast<long>(x1 + (768L*x0))];
                        auto tmp16 = in_ptr7[static_cast<long>(x1)];
                        auto tmp18 = in_ptr8[static_cast<long>(x1 + (768L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp12 = c10::convert<float>(tmp11);
                        auto tmp13 = static_cast<float>(1.1111111111111112);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = decltype(tmp10)(tmp10 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        in_out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp15;
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp19;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr2[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr9[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
                    auto tmp17 = in_ptr10[static_cast<long>(x0)];
                    auto tmp24 = in_ptr11[static_cast<long>(x0)];
                    auto tmp28 = in_ptr12[static_cast<long>(x0)];
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr7 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr9 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_161 = async_compile.cpp('''
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_16, primals_24, primals_25, primals_32, primals_38, primals_46, primals_47, primals_54, primals_60, primals_68, primals_69, primals_76, primals_82, primals_90, primals_91, primals_98, primals_104, primals_112, primals_113, primals_120, primals_126, primals_134, primals_135, primals_142, primals_148, primals_156, primals_157, primals_164, primals_170, primals_178, primals_179, primals_186, primals_192, primals_200, primals_201, primals_208, primals_214, primals_222, primals_223, primals_230, primals_236, primals_244, primals_245, primals_252, primals_258, primals_266, primals_267, primals_274, primals_280, primals_284, primals_290, primals_291, expand, slice_4, mul_1, getitem_3, view, addmm, permute_3, convolution, convolution_1, permute_9, view_9, full_default_1, unsqueeze_8, getitem_221, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_30, getitem_7, mul_4, view_32, addmm_5, view_34, getitem_11, mul_9, view_36, addmm_7, permute_22, convolution_2, convolution_3, permute_28, view_45, getitem_219, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_66, getitem_17, mul_12, view_68, addmm_12, view_70, getitem_21, mul_17, view_72, addmm_14, permute_41, convolution_4, convolution_5, permute_47, view_81, getitem_217, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_102, getitem_27, mul_20, view_104, addmm_19, view_106, getitem_31, mul_25, view_108, addmm_21, permute_60, convolution_6, convolution_7, permute_66, view_117, getitem_215, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_138, getitem_37, mul_28, view_140, addmm_26, view_142, getitem_41, mul_33, view_144, addmm_28, permute_79, convolution_8, convolution_9, permute_85, view_153, getitem_213, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_174, getitem_47, mul_36, view_176, addmm_33, view_178, getitem_51, mul_41, view_180, addmm_35, permute_98, convolution_10, convolution_11, permute_104, view_189, getitem_211, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_210, getitem_57, mul_44, view_212, addmm_40, view_214, getitem_61, mul_49, view_216, addmm_42, permute_117, convolution_12, convolution_13, permute_123, view_225, getitem_209, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_246, getitem_67, mul_52, view_248, addmm_47, view_250, getitem_71, mul_57, view_252, addmm_49, permute_136, convolution_14, convolution_15, permute_142, view_261, getitem_207, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_282, getitem_77, mul_60, view_284, addmm_54, view_286, getitem_81, mul_65, view_288, addmm_56, permute_155, convolution_16, convolution_17, permute_161, view_297, getitem_205, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_318, getitem_87, mul_68, view_320, addmm_61, view_322, getitem_91, mul_73, view_324, addmm_63, permute_174, convolution_18, convolution_19, permute_180, view_333, getitem_203, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_354, getitem_97, mul_76, view_356, addmm_68, view_358, getitem_101, mul_81, view_360, addmm_70, permute_193, convolution_20, convolution_21, permute_199, view_369, getitem_201, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_390, getitem_107, mul_84, view_392, addmm_75, view_394, getitem_111, mul_89, view_396, addmm_77, permute_212, convolution_22, convolution_23, permute_218, view_405, getitem_199, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_426, getitem_117, mul_92, view_428, addmm_82, view_430, getitem_121, mul_97, view_432, addmm_84, mul_102, view_434, sub_52, convert_element_type, permute_230, div_38, permute_234, div_39, permute_238, permute_242, div_40, permute_246, permute_256, permute_257, permute_261, alias_27, permute_275, permute_279, permute_283, div_42, permute_287, permute_291, div_43, permute_295, permute_305, permute_306, permute_310, alias_29, permute_324, permute_328, permute_332, div_45, permute_336, permute_340, div_46, permute_344, permute_354, permute_355, permute_359, alias_31, permute_373, permute_377, permute_381, div_48, permute_385, permute_389, div_49, permute_393, permute_403, permute_404, permute_408, alias_33, permute_422, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_452, permute_453, permute_457, alias_35, permute_471, permute_475, permute_479, div_54, permute_483, permute_487, div_55, permute_491, permute_501, permute_502, permute_506, alias_37, permute_520, permute_524, permute_528, div_57, permute_532, permute_536, div_58, permute_540, permute_550, permute_551, permute_555, alias_39, permute_569, permute_573, permute_577, div_60, permute_581, permute_585, div_61, permute_589, permute_599, permute_600, permute_604, alias_41, permute_618, permute_622, permute_626, div_63, permute_630, permute_634, div_64, permute_638, permute_648, permute_649, permute_653, alias_43, permute_667, permute_671, permute_675, div_66, permute_679, permute_683, div_67, permute_687, permute_697, permute_698, permute_702, alias_45, permute_716, permute_720, permute_724, div_69, permute_728, permute_732, div_70, permute_736, permute_746, permute_747, permute_751, alias_47, permute_765, permute_769, permute_773, div_72, permute_777, permute_781, div_73, permute_785, permute_795, permute_796, permute_800, alias_49, permute_814, permute_818, permute_822, div_75, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (384, 1), (1, 1))
    assert_size_stride(primals_2, (384, 1), (1, 1))
    assert_size_stride(primals_3, (384, 1), (1, 1))
    assert_size_stride(primals_4, (384, 1), (1, 1))
    assert_size_stride(primals_5, (384, 1), (1, 1))
    assert_size_stride(primals_6, (384, 1), (1, 1))
    assert_size_stride(primals_7, (384, 1), (1, 1))
    assert_size_stride(primals_8, (384, 1), (1, 1))
    assert_size_stride(primals_9, (384, 1), (1, 1))
    assert_size_stride(primals_10, (384, 1), (1, 1))
    assert_size_stride(primals_11, (384, 1), (1, 1))
    assert_size_stride(primals_12, (384, 1), (1, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_24, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_25, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_46, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_47, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_68, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_69, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_90, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_91, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_112, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_113, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_134, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_135, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_156, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_157, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_178, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_179, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_200, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_201, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_222, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_223, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_244, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_245, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_266, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_267, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_274, (768, ), (1, ))
    assert_size_stride(primals_280, (768, ), (1, ))
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_290, (1, 512), (512, 1))
    assert_size_stride(primals_291, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_4, (1, 512), (512, 1))
    assert_size_stride(mul_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(addmm, (512, 384), (384, 1))
    assert_size_stride(permute_3, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_1, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_9, (384, 54), (1, 384))
    assert_size_stride(view_9, (512, 384), (1, 512))
    assert_size_stride(full_default_1, (1, 1), (1, 1))
    assert_size_stride(unsqueeze_8, (9, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_221, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_67, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_68, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_23, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_69, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_70, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_4, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_32, (512, 768), (768, 1))
    assert_size_stride(addmm_5, (512, 3072), (3072, 1))
    assert_size_stride(view_34, (512, 3072), (3072, 1))
    assert_size_stride(getitem_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_9, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_36, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 384), (384, 1))
    assert_size_stride(permute_22, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_2, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_3, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_28, (384, 54), (1, 384))
    assert_size_stride(view_45, (512, 384), (1, 512))
    assert_size_stride(getitem_219, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_61, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_62, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_21, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_63, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_64, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(getitem_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_12, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_68, (512, 768), (768, 1))
    assert_size_stride(addmm_12, (512, 3072), (3072, 1))
    assert_size_stride(view_70, (512, 3072), (3072, 1))
    assert_size_stride(getitem_21, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_72, (512, 768), (768, 1))
    assert_size_stride(addmm_14, (512, 384), (384, 1))
    assert_size_stride(permute_41, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_4, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_5, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_47, (384, 54), (1, 384))
    assert_size_stride(view_81, (512, 384), (1, 512))
    assert_size_stride(getitem_217, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_55, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_56, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_19, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_57, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_58, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_102, (512, 768), (768, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_20, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(view_106, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_108, (512, 768), (768, 1))
    assert_size_stride(addmm_21, (512, 384), (384, 1))
    assert_size_stride(permute_60, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_6, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_7, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_66, (384, 54), (1, 384))
    assert_size_stride(view_117, (512, 384), (1, 512))
    assert_size_stride(getitem_215, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_49, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_50, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_17, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_51, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_52, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_138, (512, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_28, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_140, (512, 768), (768, 1))
    assert_size_stride(addmm_26, (512, 3072), (3072, 1))
    assert_size_stride(view_142, (512, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_33, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_144, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 384), (384, 1))
    assert_size_stride(permute_79, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_8, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_9, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_85, (384, 54), (1, 384))
    assert_size_stride(view_153, (512, 384), (1, 512))
    assert_size_stride(getitem_213, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_43, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_44, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_15, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_45, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_46, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_174, (512, 768), (768, 1))
    assert_size_stride(getitem_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(addmm_33, (512, 3072), (3072, 1))
    assert_size_stride(view_178, (512, 3072), (3072, 1))
    assert_size_stride(getitem_51, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_180, (512, 768), (768, 1))
    assert_size_stride(addmm_35, (512, 384), (384, 1))
    assert_size_stride(permute_98, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_10, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_11, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_104, (384, 54), (1, 384))
    assert_size_stride(view_189, (512, 384), (1, 512))
    assert_size_stride(getitem_211, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_37, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_38, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_13, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_39, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_40, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_210, (512, 768), (768, 1))
    assert_size_stride(getitem_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_44, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_212, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_214, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_42, (512, 384), (384, 1))
    assert_size_stride(permute_117, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_12, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_13, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_123, (384, 54), (1, 384))
    assert_size_stride(view_225, (512, 384), (1, 512))
    assert_size_stride(getitem_209, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_31, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_32, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_11, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_33, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_34, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_246, (512, 768), (768, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_52, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_248, (512, 768), (768, 1))
    assert_size_stride(addmm_47, (512, 3072), (3072, 1))
    assert_size_stride(view_250, (512, 3072), (3072, 1))
    assert_size_stride(getitem_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_252, (512, 768), (768, 1))
    assert_size_stride(addmm_49, (512, 384), (384, 1))
    assert_size_stride(permute_136, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_14, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_15, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_142, (384, 54), (1, 384))
    assert_size_stride(view_261, (512, 384), (1, 512))
    assert_size_stride(getitem_207, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_25, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_26, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_9, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_27, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_28, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_282, (512, 768), (768, 1))
    assert_size_stride(getitem_77, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_60, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_284, (512, 768), (768, 1))
    assert_size_stride(addmm_54, (512, 3072), (3072, 1))
    assert_size_stride(view_286, (512, 3072), (3072, 1))
    assert_size_stride(getitem_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_65, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_288, (512, 768), (768, 1))
    assert_size_stride(addmm_56, (512, 384), (384, 1))
    assert_size_stride(permute_155, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_16, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_17, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_161, (384, 54), (1, 384))
    assert_size_stride(view_297, (512, 384), (1, 512))
    assert_size_stride(getitem_205, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_19, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_20, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_7, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_21, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_22, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_318, (512, 768), (768, 1))
    assert_size_stride(getitem_87, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_68, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_320, (512, 768), (768, 1))
    assert_size_stride(addmm_61, (512, 3072), (3072, 1))
    assert_size_stride(view_322, (512, 3072), (3072, 1))
    assert_size_stride(getitem_91, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_324, (512, 768), (768, 1))
    assert_size_stride(addmm_63, (512, 384), (384, 1))
    assert_size_stride(permute_174, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_18, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_19, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_180, (384, 54), (1, 384))
    assert_size_stride(view_333, (512, 384), (1, 512))
    assert_size_stride(getitem_203, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_13, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_14, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_5, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_15, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_16, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_354, (512, 768), (768, 1))
    assert_size_stride(getitem_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_76, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_356, (512, 768), (768, 1))
    assert_size_stride(addmm_68, (512, 3072), (3072, 1))
    assert_size_stride(view_358, (512, 3072), (3072, 1))
    assert_size_stride(getitem_101, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_360, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 384), (384, 1))
    assert_size_stride(permute_193, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_20, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_21, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_199, (384, 54), (1, 384))
    assert_size_stride(view_369, (512, 384), (1, 512))
    assert_size_stride(getitem_201, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_7, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_8, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_3, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_9, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_10, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_390, (512, 768), (768, 1))
    assert_size_stride(getitem_107, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_84, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_392, (512, 768), (768, 1))
    assert_size_stride(addmm_75, (512, 3072), (3072, 1))
    assert_size_stride(view_394, (512, 3072), (3072, 1))
    assert_size_stride(getitem_111, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_89, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_396, (512, 768), (768, 1))
    assert_size_stride(addmm_77, (512, 384), (384, 1))
    assert_size_stride(permute_212, (1, 768, 512), (393216, 1, 768))
    assert_size_stride(convolution_22, (1, 768, 512), (393216, 512, 1))
    assert_size_stride(convolution_23, (1, 384, 512), (196608, 512, 1))
    assert_size_stride(permute_218, (384, 54), (1, 384))
    assert_size_stride(view_405, (512, 384), (1, 512))
    assert_size_stride(getitem_199, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_1, (6, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_2, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_1, (1, 6, 512, 512), (1572864, 1, 3072, 6))
    assert_size_stride(permute_default_3, (6, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_4, (6, 512, 64), (32768, 64, 1))
    assert_size_stride(view_426, (512, 768), (768, 1))
    assert_size_stride(getitem_117, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_92, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_428, (512, 768), (768, 1))
    assert_size_stride(addmm_82, (512, 3072), (3072, 1))
    assert_size_stride(view_430, (512, 3072), (3072, 1))
    assert_size_stride(getitem_121, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_432, (512, 768), (768, 1))
    assert_size_stride(addmm_84, (512, 768), (768, 1))
    assert_size_stride(mul_102, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_434, (512, 768), (768, 1))
    assert_size_stride(sub_52, (512, 30522), (30522, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_230, (30522, 768), (768, 1))
    assert_size_stride(div_38, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_234, (768, 768), (768, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_238, (768, 3072), (3072, 1))
    assert_size_stride(permute_242, (3072, 768), (768, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_246, (768, 768), (768, 1))
    assert_size_stride(permute_256, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_257, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_261, (384, 768), (768, 1))
    assert_size_stride(alias_27, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_275, (384, 768), (768, 1))
    assert_size_stride(permute_279, (384, 768), (768, 1))
    assert_size_stride(permute_283, (384, 768), (768, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_287, (768, 3072), (3072, 1))
    assert_size_stride(permute_291, (3072, 768), (768, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_295, (768, 768), (768, 1))
    assert_size_stride(permute_305, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_306, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_310, (384, 768), (768, 1))
    assert_size_stride(alias_29, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_324, (384, 768), (768, 1))
    assert_size_stride(permute_328, (384, 768), (768, 1))
    assert_size_stride(permute_332, (384, 768), (768, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_336, (768, 3072), (3072, 1))
    assert_size_stride(permute_340, (3072, 768), (768, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_344, (768, 768), (768, 1))
    assert_size_stride(permute_354, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_355, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_359, (384, 768), (768, 1))
    assert_size_stride(alias_31, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_373, (384, 768), (768, 1))
    assert_size_stride(permute_377, (384, 768), (768, 1))
    assert_size_stride(permute_381, (384, 768), (768, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_385, (768, 3072), (3072, 1))
    assert_size_stride(permute_389, (3072, 768), (768, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_403, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_404, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_408, (384, 768), (768, 1))
    assert_size_stride(alias_33, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_422, (384, 768), (768, 1))
    assert_size_stride(permute_426, (384, 768), (768, 1))
    assert_size_stride(permute_430, (384, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_434, (768, 3072), (3072, 1))
    assert_size_stride(permute_438, (3072, 768), (768, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_442, (768, 768), (768, 1))
    assert_size_stride(permute_452, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_453, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_457, (384, 768), (768, 1))
    assert_size_stride(alias_35, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_471, (384, 768), (768, 1))
    assert_size_stride(permute_475, (384, 768), (768, 1))
    assert_size_stride(permute_479, (384, 768), (768, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_483, (768, 3072), (3072, 1))
    assert_size_stride(permute_487, (3072, 768), (768, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_491, (768, 768), (768, 1))
    assert_size_stride(permute_501, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_502, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_506, (384, 768), (768, 1))
    assert_size_stride(alias_37, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_520, (384, 768), (768, 1))
    assert_size_stride(permute_524, (384, 768), (768, 1))
    assert_size_stride(permute_528, (384, 768), (768, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_532, (768, 3072), (3072, 1))
    assert_size_stride(permute_536, (3072, 768), (768, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_540, (768, 768), (768, 1))
    assert_size_stride(permute_550, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_551, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_555, (384, 768), (768, 1))
    assert_size_stride(alias_39, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_569, (384, 768), (768, 1))
    assert_size_stride(permute_573, (384, 768), (768, 1))
    assert_size_stride(permute_577, (384, 768), (768, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_581, (768, 3072), (3072, 1))
    assert_size_stride(permute_585, (3072, 768), (768, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_589, (768, 768), (768, 1))
    assert_size_stride(permute_599, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_600, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_604, (384, 768), (768, 1))
    assert_size_stride(alias_41, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_618, (384, 768), (768, 1))
    assert_size_stride(permute_622, (384, 768), (768, 1))
    assert_size_stride(permute_626, (384, 768), (768, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_630, (768, 3072), (3072, 1))
    assert_size_stride(permute_634, (3072, 768), (768, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_638, (768, 768), (768, 1))
    assert_size_stride(permute_648, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_649, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_653, (384, 768), (768, 1))
    assert_size_stride(alias_43, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_667, (384, 768), (768, 1))
    assert_size_stride(permute_671, (384, 768), (768, 1))
    assert_size_stride(permute_675, (384, 768), (768, 1))
    assert_size_stride(div_66, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_679, (768, 3072), (3072, 1))
    assert_size_stride(permute_683, (3072, 768), (768, 1))
    assert_size_stride(div_67, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_687, (768, 768), (768, 1))
    assert_size_stride(permute_697, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_698, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_702, (384, 768), (768, 1))
    assert_size_stride(alias_45, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_716, (384, 768), (768, 1))
    assert_size_stride(permute_720, (384, 768), (768, 1))
    assert_size_stride(permute_724, (384, 768), (768, 1))
    assert_size_stride(div_69, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_728, (768, 3072), (3072, 1))
    assert_size_stride(permute_732, (3072, 768), (768, 1))
    assert_size_stride(div_70, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_736, (768, 768), (768, 1))
    assert_size_stride(permute_746, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_747, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_751, (384, 768), (768, 1))
    assert_size_stride(alias_47, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_765, (384, 768), (768, 1))
    assert_size_stride(permute_769, (384, 768), (768, 1))
    assert_size_stride(permute_773, (384, 768), (768, 1))
    assert_size_stride(div_72, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_777, (768, 3072), (3072, 1))
    assert_size_stride(permute_781, (3072, 768), (768, 1))
    assert_size_stride(div_73, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_785, (768, 768), (768, 1))
    assert_size_stride(permute_795, (3072, 9, 64), (576, 1, 9))
    assert_size_stride(permute_796, (3072, 1, 9), (9, 27648, 1))
    assert_size_stride(permute_800, (384, 768), (768, 1))
    assert_size_stride(alias_49, (3072, 9, 1), (9, 1, 27648))
    assert_size_stride(permute_814, (384, 768), (768, 1))
    assert_size_stride(permute_818, (384, 768), (768, 1))
    assert_size_stride(permute_822, (384, 768), (768, 1))
    assert_size_stride(div_75, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 30522), (15627264, 30522, 1))
    buf0 = empty((512, 30522), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_291.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((512, 30522), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 512, 30522), (15627264, 30522, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_52.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del convert_element_type
    del primals_291
    del sub_52
    del tangents_1
    del tangents_2
    buf6 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (512, 30522), (30522, 1), 0), permute_230, out=buf6)
    del permute_230
    buf7 = empty((30522, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (30522, 512), (1, 30522), 0), view_434, out=buf7)
    del view_434
    buf8 = empty((1, 30522), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf4, (1, 512, 1), (512, 1, 512), 0); del buf4  # reuse
    buf10 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf11 = empty((768, ), device='cpu', dtype=torch.float32)
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf6, (1, 512, 768), (393216, 768, 1), 0); del buf6  # reuse
    cpp_fused_gelu_gelu_backward_native_layer_norm_backward_sum_2(c_void_p(buf13.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(mul_102.data_ptr()), c_void_p(div_38.data_ptr()), c_void_p(addmm_84.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del addmm_84
    del buf5
    del div_38
    del mul_102
    del primals_284
    buf14 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_234, out=buf14)
    del permute_234
    buf15 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (768, 512), (1, 768), 0), view_432, out=buf15)
    del view_432
    buf16 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf17 = buf9; del buf9  # reuse
    buf18 = buf10; del buf10  # reuse
    buf19 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf20 = empty((768, ), device='cpu', dtype=torch.float32)
    buf21 = empty((768, ), device='cpu', dtype=torch.float32)
    buf22 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(mul_97.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(getitem_121.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del div_39
    del getitem_121
    del mul_97
    del primals_280
    buf23 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (512, 768), (768, 1), 0), permute_238, out=buf23)
    del permute_238
    buf24 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (768, 512), (1, 768), 0), view_430, out=buf24)
    del view_430
    buf25 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf23, (1, 512, 3072), (1572864, 3072, 1), 0); del buf23  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf26.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(addmm_82.data_ptr()), c_void_p(buf25.data_ptr()))
    del addmm_82
    buf27 = reinterpret_tensor(buf22, (512, 768), (768, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0), permute_242, out=buf27)
    del permute_242
    buf28 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (3072, 512), (1, 3072), 0), view_428, out=buf28)
    del view_428
    buf29 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf30 = buf18; del buf18  # reuse
    buf31 = buf17; del buf17  # reuse
    buf32 = reinterpret_tensor(buf14, (1, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
    buf33 = empty((768, ), device='cpu', dtype=torch.float32)
    buf34 = empty((768, ), device='cpu', dtype=torch.float32)
    buf35 = buf13; del buf13  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_5(c_void_p(buf26.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(mul_92.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del div_40
    del getitem_117
    del mul_92
    del primals_274
    buf36 = buf27; del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (512, 768), (768, 1), 0), permute_246, out=buf36)
    del permute_246
    buf37 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (768, 512), (1, 768), 0), view_426, out=buf37)
    del view_426
    buf38 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_6(c_void_p(buf35.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_1, reinterpret_tensor(buf36, (6, 512, 64), (64, 768, 1), 0), out=buf39)
    del permute_default_1
    buf40 = reinterpret_tensor(buf26, (6, 512, 512), (262144, 512, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf36, (6, 512, 64), (64, 768, 1), 0), permute_default_2, out=buf40)
    del permute_default_2
    buf41 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf42 = reinterpret_tensor(buf40, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf40  # reuse
    cpp_fused_7(c_void_p(buf42.data_ptr()), c_void_p(getitem_199.data_ptr()), c_void_p(alias_default_1.data_ptr()), c_void_p(buf41.data_ptr()))
    del alias_default_1
    del getitem_199
    buf43 = empty((6, 64, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_3, reinterpret_tensor(buf42, (6, 512, 512), (262144, 512, 1), 0), out=buf43)
    del permute_default_3
    buf44 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf42, (6, 512, 512), (262144, 512, 1), 0), permute_default_4, out=buf44)
    del permute_default_4
    buf45 = empty((512, 384), device='cpu', dtype=torch.float32)
    cpp_fused_clone_8(c_void_p(buf36.data_ptr()), c_void_p(buf45.data_ptr()))
    buf46 = empty((3072, 9, 1), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_256, reinterpret_tensor(buf45, (3072, 64, 1), (64, 1, 0), 0), out=buf46)
    del permute_256
    buf47 = empty((3072, 64, 9), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf45, (3072, 64, 1), (64, 1, 0), 0), permute_257, out=buf47)
    del permute_257
    buf48 = empty((1, 384, 520, 1), device='cpu', dtype=torch.float32)
    cpp_fused_col2im_9(c_void_p(buf48.data_ptr()))
    aten.index_put_(buf48, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf47, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf51 = reinterpret_tensor(buf45, (512, 384), (1, 512), 0); del buf45  # reuse
    buf54 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_10(c_void_p(buf48.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf54.data_ptr()))
    buf52 = buf36; del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf51, permute_261, out=buf52)
    del permute_261
    buf53 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (384, 512), (512, 1), 0), view_396, out=buf53)
    buf55 = reinterpret_tensor(buf41, (3072, 1, 1), (1, 3072, 3072), 0); del buf41  # reuse
    buf56 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf57 = buf46; del buf46  # reuse
    cpp_fused__softmax_backward_data_sum_11(c_void_p(buf57.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del alias_27
    buf58 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (54, 512), (1, 54), 0), view_405, out=buf58)
    del view_405
    buf59 = reinterpret_tensor(buf51, (384, 512), (512, 1), 0); del buf51  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_218, reinterpret_tensor(buf57, (54, 512), (1, 54), 0), out=buf59)
    del permute_218
    buf60 = empty((1, 512, 384), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    buf61 = empty((1, 384, 512), device='cpu', dtype=torch.float32)
    buf62 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_12(c_void_p(buf59.data_ptr()), c_void_p(addmm_77.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del addmm_77
    del convolution_23
    del primals_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf63 = aten.convolution_backward(buf61, convolution_22, primals_267, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_22
    del primals_267
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf66 = aten.convolution_backward(buf64, permute_212, primals_266, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_212
    del primals_266
    buf67 = buf66[0]
    buf68 = buf66[1]
    del buf66
    buf69 = reinterpret_tensor(buf61, (512, 384), (384, 1), 0); del buf61  # reuse
    cpp_fused_view_13(c_void_p(buf39.data_ptr()), c_void_p(buf69.data_ptr()))
    buf70 = reinterpret_tensor(buf64, (512, 768), (768, 1), 0); del buf64  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf69, permute_275, out=buf70)
    del permute_275
    buf71 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf69, (384, 512), (1, 384), 0), view_396, out=buf71)
    buf72 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf73 = reinterpret_tensor(buf43, (512, 384), (1, 512), 0); del buf43  # reuse
    cpp_fused_sum_view_14(c_void_p(buf73.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf72.data_ptr()))
    buf74 = reinterpret_tensor(buf35, (512, 768), (768, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf73, permute_279, out=buf74)
    del permute_279
    buf75 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (384, 512), (512, 1), 0), view_396, out=buf75)
    buf76 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_15(c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()))
    buf79 = reinterpret_tensor(buf19, (512, 768), (768, 1), 0); del buf19  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf78, permute_283, out=buf79)
    del permute_283
    buf80 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (384, 512), (512, 1), 0), view_396, out=buf80)
    del view_396
    buf81 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf82 = buf32; del buf32  # reuse
    buf83 = buf31; del buf31  # reuse
    buf84 = buf30; del buf30  # reuse
    buf85 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf86 = empty((768, ), device='cpu', dtype=torch.float32)
    buf87 = empty((768, ), device='cpu', dtype=torch.float32)
    buf88 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_16(c_void_p(buf82.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(mul_89.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    del buf52
    del buf67
    del div_42
    del getitem_111
    del mul_89
    del primals_258
    buf89 = reinterpret_tensor(buf42, (512, 3072), (3072, 1), 0); del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf88, (512, 768), (768, 1), 0), permute_287, out=buf89)
    del permute_287
    buf90 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf88, (768, 512), (1, 768), 0), view_394, out=buf90)
    del view_394
    buf91 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf92 = reinterpret_tensor(buf89, (1, 512, 3072), (1572864, 3072, 1), 0); del buf89  # reuse
    cpp_fused_gelu_gelu_backward_sum_17(c_void_p(buf92.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(addmm_75.data_ptr()), c_void_p(buf91.data_ptr()))
    del addmm_75
    buf93 = reinterpret_tensor(buf88, (512, 768), (768, 1), 0); del buf88  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf92, (512, 3072), (3072, 1), 0), permute_291, out=buf93)
    del permute_291
    buf94 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf92, (3072, 512), (1, 3072), 0), view_392, out=buf94)
    del view_392
    buf95 = reinterpret_tensor(buf55, (1, 3072), (3072, 1), 0); del buf55  # reuse
    buf96 = buf84; del buf84  # reuse
    buf97 = buf83; del buf83  # reuse
    buf98 = buf82; del buf82  # reuse
    buf99 = empty((768, ), device='cpu', dtype=torch.float32)
    buf100 = empty((768, ), device='cpu', dtype=torch.float32)
    buf101 = reinterpret_tensor(buf79, (1, 512, 768), (393216, 768, 1), 0); del buf79  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_18(c_void_p(buf92.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(mul_84.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del div_43
    del getitem_107
    del mul_84
    del primals_252
    buf102 = buf93; del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (512, 768), (768, 1), 0), permute_295, out=buf102)
    del permute_295
    buf103 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (768, 512), (1, 768), 0), view_390, out=buf103)
    del view_390
    buf104 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_19(c_void_p(buf101.data_ptr()), c_void_p(buf104.data_ptr()))
    buf105 = reinterpret_tensor(buf78, (6, 512, 64), (32768, 64, 1), 0); del buf78  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_7, reinterpret_tensor(buf102, (6, 512, 64), (64, 768, 1), 0), out=buf105)
    del permute_default_7
    buf106 = reinterpret_tensor(buf92, (6, 512, 512), (262144, 512, 1), 0); del buf92  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf102, (6, 512, 64), (64, 768, 1), 0), permute_default_8, out=buf106)
    del permute_default_8
    buf107 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf106, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf106  # reuse
    cpp_fused_20(c_void_p(buf108.data_ptr()), c_void_p(getitem_201.data_ptr()), c_void_p(alias_default_3.data_ptr()), c_void_p(buf107.data_ptr()))
    del alias_default_3
    del getitem_201
    buf109 = reinterpret_tensor(buf73, (6, 64, 512), (32768, 512, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_9, reinterpret_tensor(buf108, (6, 512, 512), (262144, 512, 1), 0), out=buf109)
    del permute_default_9
    buf110 = reinterpret_tensor(buf69, (6, 512, 64), (32768, 64, 1), 0); del buf69  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf108, (6, 512, 512), (262144, 512, 1), 0), permute_default_10, out=buf110)
    del permute_default_10
    buf111 = reinterpret_tensor(buf39, (512, 384), (384, 1), 0); del buf39  # reuse
    cpp_fused_clone_21(c_void_p(buf102.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = buf57; del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_305, reinterpret_tensor(buf111, (3072, 64, 1), (64, 1, 0), 0), out=buf112)
    del permute_305
    buf113 = buf47; del buf47  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf111, (3072, 64, 1), (64, 1, 0), 0), permute_306, out=buf113)
    del permute_306
    buf114 = buf48; del buf48  # reuse
    cpp_fused_col2im_22(c_void_p(buf114.data_ptr()))
    aten.index_put_(buf114, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf113, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf117 = reinterpret_tensor(buf111, (512, 384), (1, 512), 0); del buf111  # reuse
    buf120 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_23(c_void_p(buf114.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf120.data_ptr()))
    buf118 = buf102; del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf117, permute_310, out=buf118)
    del permute_310
    buf119 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (384, 512), (512, 1), 0), view_360, out=buf119)
    buf121 = reinterpret_tensor(buf107, (3072, 1, 1), (1, 3072, 3072), 0); del buf107  # reuse
    buf122 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf123 = buf112; del buf112  # reuse
    cpp_fused__softmax_backward_data_sum_24(c_void_p(buf123.data_ptr()), c_void_p(alias_29.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del alias_29
    buf124 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (54, 512), (1, 54), 0), view_369, out=buf124)
    del view_369
    buf125 = reinterpret_tensor(buf117, (384, 512), (512, 1), 0); del buf117  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_199, reinterpret_tensor(buf123, (54, 512), (1, 54), 0), out=buf125)
    del permute_199
    buf126 = buf60; del buf60  # reuse
    buf144 = reinterpret_tensor(buf59, (512, 384), (1, 512), 0); del buf59  # reuse
    buf127 = reinterpret_tensor(buf44, (1, 384, 512), (196608, 512, 1), 0); del buf44  # reuse
    buf128 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_25(c_void_p(buf125.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del addmm_70
    del convolution_21
    del primals_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf129 = aten.convolution_backward(buf127, convolution_20, primals_245, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_20
    del primals_245
    buf130 = buf129[0]
    buf131 = buf129[1]
    del buf129
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf132 = aten.convolution_backward(buf130, permute_193, primals_244, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_193
    del primals_244
    buf133 = buf132[0]
    buf134 = buf132[1]
    del buf132
    buf135 = reinterpret_tensor(buf127, (512, 384), (384, 1), 0); del buf127  # reuse
    cpp_fused_view_26(c_void_p(buf105.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = reinterpret_tensor(buf130, (512, 768), (768, 1), 0); del buf130  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf135, permute_324, out=buf136)
    del permute_324
    buf137 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (384, 512), (1, 384), 0), view_360, out=buf137)
    buf138 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf139 = reinterpret_tensor(buf109, (512, 384), (1, 512), 0); del buf109  # reuse
    cpp_fused_sum_view_27(c_void_p(buf139.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf138.data_ptr()))
    buf140 = reinterpret_tensor(buf101, (512, 768), (768, 1), 0); del buf101  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf139, permute_328, out=buf140)
    del permute_328
    buf141 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (384, 512), (512, 1), 0), view_360, out=buf141)
    buf142 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()))
    buf145 = reinterpret_tensor(buf85, (512, 768), (768, 1), 0); del buf85  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf144, permute_332, out=buf145)
    del permute_332
    buf146 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf144, (384, 512), (512, 1), 0), view_360, out=buf146)
    del view_360
    buf147 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf148 = reinterpret_tensor(buf118, (1, 512, 768), (393216, 768, 1), 0); del buf118  # reuse
    buf149 = buf97; del buf97  # reuse
    buf150 = buf96; del buf96  # reuse
    buf151 = reinterpret_tensor(buf74, (1, 512, 768), (393216, 768, 1), 0); del buf74  # reuse
    buf152 = empty((768, ), device='cpu', dtype=torch.float32)
    buf153 = empty((768, ), device='cpu', dtype=torch.float32)
    buf154 = reinterpret_tensor(buf70, (1, 512, 768), (393216, 768, 1), 0); del buf70  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_29(c_void_p(buf148.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(mul_81.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del buf133
    del buf136
    del div_45
    del getitem_101
    del mul_81
    del primals_236
    buf155 = reinterpret_tensor(buf108, (512, 3072), (3072, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (512, 768), (768, 1), 0), permute_336, out=buf155)
    del permute_336
    buf156 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (768, 512), (1, 768), 0), view_358, out=buf156)
    del view_358
    buf157 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf158 = reinterpret_tensor(buf155, (1, 512, 3072), (1572864, 3072, 1), 0); del buf155  # reuse
    cpp_fused_gelu_gelu_backward_sum_30(c_void_p(buf158.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(addmm_68.data_ptr()), c_void_p(buf157.data_ptr()))
    del addmm_68
    buf159 = reinterpret_tensor(buf154, (512, 768), (768, 1), 0); del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (512, 3072), (3072, 1), 0), permute_340, out=buf159)
    del permute_340
    buf160 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (3072, 512), (1, 3072), 0), view_356, out=buf160)
    del view_356
    buf161 = reinterpret_tensor(buf121, (1, 3072), (3072, 1), 0); del buf121  # reuse
    buf162 = buf150; del buf150  # reuse
    buf163 = buf149; del buf149  # reuse
    buf164 = buf98; del buf98  # reuse
    buf165 = empty((768, ), device='cpu', dtype=torch.float32)
    buf166 = empty((768, ), device='cpu', dtype=torch.float32)
    buf167 = buf148; del buf148  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_31(c_void_p(buf158.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(mul_76.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del div_46
    del getitem_97
    del mul_76
    del primals_230
    buf168 = buf159; del buf159  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (512, 768), (768, 1), 0), permute_344, out=buf168)
    del permute_344
    buf169 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (768, 512), (1, 768), 0), view_354, out=buf169)
    del view_354
    buf170 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_32(c_void_p(buf167.data_ptr()), c_void_p(buf170.data_ptr()))
    buf171 = reinterpret_tensor(buf144, (6, 512, 64), (32768, 64, 1), 0); del buf144  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_13, reinterpret_tensor(buf168, (6, 512, 64), (64, 768, 1), 0), out=buf171)
    del permute_default_13
    buf172 = reinterpret_tensor(buf158, (6, 512, 512), (262144, 512, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf168, (6, 512, 64), (64, 768, 1), 0), permute_default_14, out=buf172)
    del permute_default_14
    buf173 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf174 = reinterpret_tensor(buf172, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf172  # reuse
    cpp_fused_33(c_void_p(buf174.data_ptr()), c_void_p(getitem_203.data_ptr()), c_void_p(alias_default_5.data_ptr()), c_void_p(buf173.data_ptr()))
    del alias_default_5
    del getitem_203
    buf175 = reinterpret_tensor(buf139, (6, 64, 512), (32768, 512, 1), 0); del buf139  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_15, reinterpret_tensor(buf174, (6, 512, 512), (262144, 512, 1), 0), out=buf175)
    del permute_default_15
    buf176 = reinterpret_tensor(buf135, (6, 512, 64), (32768, 64, 1), 0); del buf135  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf174, (6, 512, 512), (262144, 512, 1), 0), permute_default_16, out=buf176)
    del permute_default_16
    buf177 = reinterpret_tensor(buf105, (512, 384), (384, 1), 0); del buf105  # reuse
    cpp_fused_clone_34(c_void_p(buf168.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = buf123; del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_354, reinterpret_tensor(buf177, (3072, 64, 1), (64, 1, 0), 0), out=buf178)
    del permute_354
    buf179 = buf113; del buf113  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf177, (3072, 64, 1), (64, 1, 0), 0), permute_355, out=buf179)
    del permute_355
    buf180 = buf114; del buf114  # reuse
    cpp_fused_col2im_35(c_void_p(buf180.data_ptr()))
    aten.index_put_(buf180, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf179, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf183 = reinterpret_tensor(buf177, (512, 384), (1, 512), 0); del buf177  # reuse
    buf186 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_36(c_void_p(buf180.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf186.data_ptr()))
    buf184 = buf168; del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf183, permute_359, out=buf184)
    del permute_359
    buf185 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (384, 512), (512, 1), 0), view_324, out=buf185)
    buf187 = reinterpret_tensor(buf173, (3072, 1, 1), (1, 3072, 3072), 0); del buf173  # reuse
    buf188 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf189 = buf178; del buf178  # reuse
    cpp_fused__softmax_backward_data_sum_37(c_void_p(buf189.data_ptr()), c_void_p(alias_31.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del alias_31
    buf190 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (54, 512), (1, 54), 0), view_333, out=buf190)
    del view_333
    buf191 = reinterpret_tensor(buf183, (384, 512), (512, 1), 0); del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_180, reinterpret_tensor(buf189, (54, 512), (1, 54), 0), out=buf191)
    del permute_180
    buf192 = buf126; del buf126  # reuse
    buf210 = reinterpret_tensor(buf125, (512, 384), (1, 512), 0); del buf125  # reuse
    buf193 = reinterpret_tensor(buf110, (1, 384, 512), (196608, 512, 1), 0); del buf110  # reuse
    buf194 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_38(c_void_p(buf191.data_ptr()), c_void_p(addmm_63.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del addmm_63
    del convolution_19
    del primals_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf195 = aten.convolution_backward(buf193, convolution_18, primals_223, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_18
    del primals_223
    buf196 = buf195[0]
    buf197 = buf195[1]
    del buf195
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf198 = aten.convolution_backward(buf196, permute_174, primals_222, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_174
    del primals_222
    buf199 = buf198[0]
    buf200 = buf198[1]
    del buf198
    buf201 = reinterpret_tensor(buf193, (512, 384), (384, 1), 0); del buf193  # reuse
    cpp_fused_view_39(c_void_p(buf171.data_ptr()), c_void_p(buf201.data_ptr()))
    buf202 = reinterpret_tensor(buf196, (512, 768), (768, 1), 0); del buf196  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf201, permute_373, out=buf202)
    del permute_373
    buf203 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (384, 512), (1, 384), 0), view_324, out=buf203)
    buf204 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf205 = reinterpret_tensor(buf175, (512, 384), (1, 512), 0); del buf175  # reuse
    cpp_fused_sum_view_40(c_void_p(buf205.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf204.data_ptr()))
    buf206 = reinterpret_tensor(buf167, (512, 768), (768, 1), 0); del buf167  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf205, permute_377, out=buf206)
    del permute_377
    buf207 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf205, (384, 512), (512, 1), 0), view_324, out=buf207)
    buf208 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_41(c_void_p(buf205.data_ptr()), c_void_p(buf208.data_ptr()))
    buf211 = reinterpret_tensor(buf151, (512, 768), (768, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf210, permute_381, out=buf211)
    del permute_381
    buf212 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (384, 512), (512, 1), 0), view_324, out=buf212)
    del view_324
    buf213 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf214 = buf164; del buf164  # reuse
    buf215 = buf163; del buf163  # reuse
    buf216 = buf162; del buf162  # reuse
    buf217 = reinterpret_tensor(buf145, (1, 512, 768), (393216, 768, 1), 0); del buf145  # reuse
    buf218 = empty((768, ), device='cpu', dtype=torch.float32)
    buf219 = empty((768, ), device='cpu', dtype=torch.float32)
    buf220 = reinterpret_tensor(buf140, (1, 512, 768), (393216, 768, 1), 0); del buf140  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_42(c_void_p(buf214.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    del buf184
    del buf199
    del div_48
    del getitem_91
    del mul_73
    del primals_214
    buf221 = reinterpret_tensor(buf174, (512, 3072), (3072, 1), 0); del buf174  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (512, 768), (768, 1), 0), permute_385, out=buf221)
    del permute_385
    buf222 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (768, 512), (1, 768), 0), view_322, out=buf222)
    del view_322
    buf223 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf224 = reinterpret_tensor(buf221, (1, 512, 3072), (1572864, 3072, 1), 0); del buf221  # reuse
    cpp_fused_gelu_gelu_backward_sum_43(c_void_p(buf224.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(addmm_61.data_ptr()), c_void_p(buf223.data_ptr()))
    del addmm_61
    buf225 = reinterpret_tensor(buf220, (512, 768), (768, 1), 0); del buf220  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (512, 3072), (3072, 1), 0), permute_389, out=buf225)
    del permute_389
    buf226 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (3072, 512), (1, 3072), 0), view_320, out=buf226)
    del view_320
    buf227 = reinterpret_tensor(buf187, (1, 3072), (3072, 1), 0); del buf187  # reuse
    buf228 = buf216; del buf216  # reuse
    buf229 = buf215; del buf215  # reuse
    buf230 = buf214; del buf214  # reuse
    buf231 = empty((768, ), device='cpu', dtype=torch.float32)
    buf232 = empty((768, ), device='cpu', dtype=torch.float32)
    buf233 = reinterpret_tensor(buf211, (1, 512, 768), (393216, 768, 1), 0); del buf211  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_44(c_void_p(buf224.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(mul_68.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    del div_49
    del getitem_87
    del mul_68
    del primals_208
    buf234 = buf225; del buf225  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (512, 768), (768, 1), 0), permute_393, out=buf234)
    del permute_393
    buf235 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (768, 512), (1, 768), 0), view_318, out=buf235)
    del view_318
    buf236 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_45(c_void_p(buf233.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf210, (6, 512, 64), (32768, 64, 1), 0); del buf210  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_19, reinterpret_tensor(buf234, (6, 512, 64), (64, 768, 1), 0), out=buf237)
    del permute_default_19
    buf238 = reinterpret_tensor(buf224, (6, 512, 512), (262144, 512, 1), 0); del buf224  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf234, (6, 512, 64), (64, 768, 1), 0), permute_default_20, out=buf238)
    del permute_default_20
    buf239 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf240 = reinterpret_tensor(buf238, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf238  # reuse
    cpp_fused_46(c_void_p(buf240.data_ptr()), c_void_p(getitem_205.data_ptr()), c_void_p(alias_default_7.data_ptr()), c_void_p(buf239.data_ptr()))
    del alias_default_7
    del getitem_205
    buf241 = reinterpret_tensor(buf205, (6, 64, 512), (32768, 512, 1), 0); del buf205  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_21, reinterpret_tensor(buf240, (6, 512, 512), (262144, 512, 1), 0), out=buf241)
    del permute_default_21
    buf242 = reinterpret_tensor(buf201, (6, 512, 64), (32768, 64, 1), 0); del buf201  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf240, (6, 512, 512), (262144, 512, 1), 0), permute_default_22, out=buf242)
    del permute_default_22
    buf243 = reinterpret_tensor(buf171, (512, 384), (384, 1), 0); del buf171  # reuse
    cpp_fused_clone_47(c_void_p(buf234.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = buf189; del buf189  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_403, reinterpret_tensor(buf243, (3072, 64, 1), (64, 1, 0), 0), out=buf244)
    del permute_403
    buf245 = buf179; del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf243, (3072, 64, 1), (64, 1, 0), 0), permute_404, out=buf245)
    del permute_404
    buf246 = buf180; del buf180  # reuse
    cpp_fused_col2im_48(c_void_p(buf246.data_ptr()))
    aten.index_put_(buf246, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf245, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf249 = reinterpret_tensor(buf243, (512, 384), (1, 512), 0); del buf243  # reuse
    buf252 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_49(c_void_p(buf246.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf252.data_ptr()))
    buf250 = buf234; del buf234  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf249, permute_408, out=buf250)
    del permute_408
    buf251 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (384, 512), (512, 1), 0), view_288, out=buf251)
    buf253 = reinterpret_tensor(buf239, (3072, 1, 1), (1, 3072, 3072), 0); del buf239  # reuse
    buf254 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf255 = buf244; del buf244  # reuse
    cpp_fused__softmax_backward_data_sum_50(c_void_p(buf255.data_ptr()), c_void_p(alias_33.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del alias_33
    buf256 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf255, (54, 512), (1, 54), 0), view_297, out=buf256)
    del view_297
    buf257 = reinterpret_tensor(buf249, (384, 512), (512, 1), 0); del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_161, reinterpret_tensor(buf255, (54, 512), (1, 54), 0), out=buf257)
    del permute_161
    buf258 = buf192; del buf192  # reuse
    buf276 = reinterpret_tensor(buf191, (512, 384), (1, 512), 0); del buf191  # reuse
    buf259 = reinterpret_tensor(buf176, (1, 384, 512), (196608, 512, 1), 0); del buf176  # reuse
    buf260 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_51(c_void_p(buf257.data_ptr()), c_void_p(addmm_56.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del addmm_56
    del convolution_17
    del primals_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf261 = aten.convolution_backward(buf259, convolution_16, primals_201, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_16
    del primals_201
    buf262 = buf261[0]
    buf263 = buf261[1]
    del buf261
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf264 = aten.convolution_backward(buf262, permute_155, primals_200, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_155
    del primals_200
    buf265 = buf264[0]
    buf266 = buf264[1]
    del buf264
    buf267 = reinterpret_tensor(buf259, (512, 384), (384, 1), 0); del buf259  # reuse
    cpp_fused_view_52(c_void_p(buf237.data_ptr()), c_void_p(buf267.data_ptr()))
    buf268 = reinterpret_tensor(buf262, (512, 768), (768, 1), 0); del buf262  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf267, permute_422, out=buf268)
    del permute_422
    buf269 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (384, 512), (1, 384), 0), view_288, out=buf269)
    buf270 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf271 = reinterpret_tensor(buf241, (512, 384), (1, 512), 0); del buf241  # reuse
    cpp_fused_sum_view_53(c_void_p(buf271.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf270.data_ptr()))
    buf272 = reinterpret_tensor(buf233, (512, 768), (768, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf271, permute_426, out=buf272)
    del permute_426
    buf273 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (384, 512), (512, 1), 0), view_288, out=buf273)
    buf274 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf271.data_ptr()), c_void_p(buf274.data_ptr()))
    buf277 = reinterpret_tensor(buf217, (512, 768), (768, 1), 0); del buf217  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf276, permute_430, out=buf277)
    del permute_430
    buf278 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (384, 512), (512, 1), 0), view_288, out=buf278)
    del view_288
    buf279 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf280 = buf230; del buf230  # reuse
    buf281 = buf229; del buf229  # reuse
    buf282 = buf228; del buf228  # reuse
    buf283 = reinterpret_tensor(buf206, (1, 512, 768), (393216, 768, 1), 0); del buf206  # reuse
    buf284 = empty((768, ), device='cpu', dtype=torch.float32)
    buf285 = empty((768, ), device='cpu', dtype=torch.float32)
    buf286 = reinterpret_tensor(buf202, (1, 512, 768), (393216, 768, 1), 0); del buf202  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_55(c_void_p(buf280.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del buf250
    del buf265
    del div_51
    del getitem_81
    del mul_65
    del primals_192
    buf287 = reinterpret_tensor(buf240, (512, 3072), (3072, 1), 0); del buf240  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (512, 768), (768, 1), 0), permute_434, out=buf287)
    del permute_434
    buf288 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (768, 512), (1, 768), 0), view_286, out=buf288)
    del view_286
    buf289 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf290 = reinterpret_tensor(buf287, (1, 512, 3072), (1572864, 3072, 1), 0); del buf287  # reuse
    cpp_fused_gelu_gelu_backward_sum_56(c_void_p(buf290.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(addmm_54.data_ptr()), c_void_p(buf289.data_ptr()))
    del addmm_54
    buf291 = reinterpret_tensor(buf286, (512, 768), (768, 1), 0); del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf290, (512, 3072), (3072, 1), 0), permute_438, out=buf291)
    del permute_438
    buf292 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf290, (3072, 512), (1, 3072), 0), view_284, out=buf292)
    del view_284
    buf293 = reinterpret_tensor(buf253, (1, 3072), (3072, 1), 0); del buf253  # reuse
    buf294 = buf282; del buf282  # reuse
    buf295 = buf281; del buf281  # reuse
    buf296 = buf280; del buf280  # reuse
    buf297 = empty((768, ), device='cpu', dtype=torch.float32)
    buf298 = empty((768, ), device='cpu', dtype=torch.float32)
    buf299 = reinterpret_tensor(buf277, (1, 512, 768), (393216, 768, 1), 0); del buf277  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57(c_void_p(buf290.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(mul_60.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()))
    del div_52
    del getitem_77
    del mul_60
    del primals_186
    buf300 = buf291; del buf291  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (512, 768), (768, 1), 0), permute_442, out=buf300)
    del permute_442
    buf301 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (768, 512), (1, 768), 0), view_282, out=buf301)
    del view_282
    buf302 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_58(c_void_p(buf299.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf276, (6, 512, 64), (32768, 64, 1), 0); del buf276  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_25, reinterpret_tensor(buf300, (6, 512, 64), (64, 768, 1), 0), out=buf303)
    del permute_default_25
    buf304 = reinterpret_tensor(buf290, (6, 512, 512), (262144, 512, 1), 0); del buf290  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf300, (6, 512, 64), (64, 768, 1), 0), permute_default_26, out=buf304)
    del permute_default_26
    buf305 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf306 = reinterpret_tensor(buf304, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf304  # reuse
    cpp_fused_59(c_void_p(buf306.data_ptr()), c_void_p(getitem_207.data_ptr()), c_void_p(alias_default_9.data_ptr()), c_void_p(buf305.data_ptr()))
    del alias_default_9
    del getitem_207
    buf307 = reinterpret_tensor(buf271, (6, 64, 512), (32768, 512, 1), 0); del buf271  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_27, reinterpret_tensor(buf306, (6, 512, 512), (262144, 512, 1), 0), out=buf307)
    del permute_default_27
    buf308 = reinterpret_tensor(buf267, (6, 512, 64), (32768, 64, 1), 0); del buf267  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf306, (6, 512, 512), (262144, 512, 1), 0), permute_default_28, out=buf308)
    del permute_default_28
    buf309 = reinterpret_tensor(buf237, (512, 384), (384, 1), 0); del buf237  # reuse
    cpp_fused_clone_60(c_void_p(buf300.data_ptr()), c_void_p(buf309.data_ptr()))
    buf310 = buf255; del buf255  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_452, reinterpret_tensor(buf309, (3072, 64, 1), (64, 1, 0), 0), out=buf310)
    del permute_452
    buf311 = buf245; del buf245  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf309, (3072, 64, 1), (64, 1, 0), 0), permute_453, out=buf311)
    del permute_453
    buf312 = buf246; del buf246  # reuse
    cpp_fused_col2im_61(c_void_p(buf312.data_ptr()))
    aten.index_put_(buf312, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf311, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf315 = reinterpret_tensor(buf309, (512, 384), (1, 512), 0); del buf309  # reuse
    buf318 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_62(c_void_p(buf312.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf318.data_ptr()))
    buf316 = buf300; del buf300  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf315, permute_457, out=buf316)
    del permute_457
    buf317 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf315, (384, 512), (512, 1), 0), view_252, out=buf317)
    buf319 = reinterpret_tensor(buf305, (3072, 1, 1), (1, 3072, 3072), 0); del buf305  # reuse
    buf320 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf321 = buf310; del buf310  # reuse
    cpp_fused__softmax_backward_data_sum_63(c_void_p(buf321.data_ptr()), c_void_p(alias_35.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del alias_35
    buf322 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (54, 512), (1, 54), 0), view_261, out=buf322)
    del view_261
    buf323 = reinterpret_tensor(buf315, (384, 512), (512, 1), 0); del buf315  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_142, reinterpret_tensor(buf321, (54, 512), (1, 54), 0), out=buf323)
    del permute_142
    buf324 = buf258; del buf258  # reuse
    buf342 = reinterpret_tensor(buf257, (512, 384), (1, 512), 0); del buf257  # reuse
    buf325 = reinterpret_tensor(buf242, (1, 384, 512), (196608, 512, 1), 0); del buf242  # reuse
    buf326 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_64(c_void_p(buf323.data_ptr()), c_void_p(addmm_49.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del addmm_49
    del convolution_15
    del primals_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf327 = aten.convolution_backward(buf325, convolution_14, primals_179, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_14
    del primals_179
    buf328 = buf327[0]
    buf329 = buf327[1]
    del buf327
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf330 = aten.convolution_backward(buf328, permute_136, primals_178, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_136
    del primals_178
    buf331 = buf330[0]
    buf332 = buf330[1]
    del buf330
    buf333 = reinterpret_tensor(buf325, (512, 384), (384, 1), 0); del buf325  # reuse
    cpp_fused_view_65(c_void_p(buf303.data_ptr()), c_void_p(buf333.data_ptr()))
    buf334 = reinterpret_tensor(buf328, (512, 768), (768, 1), 0); del buf328  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf333, permute_471, out=buf334)
    del permute_471
    buf335 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf333, (384, 512), (1, 384), 0), view_252, out=buf335)
    buf336 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf337 = reinterpret_tensor(buf307, (512, 384), (1, 512), 0); del buf307  # reuse
    cpp_fused_sum_view_66(c_void_p(buf337.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf336.data_ptr()))
    buf338 = reinterpret_tensor(buf299, (512, 768), (768, 1), 0); del buf299  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf337, permute_475, out=buf338)
    del permute_475
    buf339 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (384, 512), (512, 1), 0), view_252, out=buf339)
    buf340 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_67(c_void_p(buf337.data_ptr()), c_void_p(buf340.data_ptr()))
    buf343 = reinterpret_tensor(buf283, (512, 768), (768, 1), 0); del buf283  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf342, permute_479, out=buf343)
    del permute_479
    buf344 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf342, (384, 512), (512, 1), 0), view_252, out=buf344)
    del view_252
    buf345 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf346 = buf296; del buf296  # reuse
    buf347 = buf295; del buf295  # reuse
    buf348 = buf294; del buf294  # reuse
    buf349 = reinterpret_tensor(buf272, (1, 512, 768), (393216, 768, 1), 0); del buf272  # reuse
    buf350 = empty((768, ), device='cpu', dtype=torch.float32)
    buf351 = empty((768, ), device='cpu', dtype=torch.float32)
    buf352 = reinterpret_tensor(buf268, (1, 512, 768), (393216, 768, 1), 0); del buf268  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_68(c_void_p(buf346.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    del buf316
    del buf331
    del div_54
    del getitem_71
    del mul_57
    del primals_170
    buf353 = reinterpret_tensor(buf306, (512, 3072), (3072, 1), 0); del buf306  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (512, 768), (768, 1), 0), permute_483, out=buf353)
    del permute_483
    buf354 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (768, 512), (1, 768), 0), view_250, out=buf354)
    del view_250
    buf355 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf356 = reinterpret_tensor(buf353, (1, 512, 3072), (1572864, 3072, 1), 0); del buf353  # reuse
    cpp_fused_gelu_gelu_backward_sum_69(c_void_p(buf356.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(addmm_47.data_ptr()), c_void_p(buf355.data_ptr()))
    del addmm_47
    buf357 = reinterpret_tensor(buf352, (512, 768), (768, 1), 0); del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf356, (512, 3072), (3072, 1), 0), permute_487, out=buf357)
    del permute_487
    buf358 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf356, (3072, 512), (1, 3072), 0), view_248, out=buf358)
    del view_248
    buf359 = reinterpret_tensor(buf319, (1, 3072), (3072, 1), 0); del buf319  # reuse
    buf360 = buf348; del buf348  # reuse
    buf361 = buf347; del buf347  # reuse
    buf362 = buf346; del buf346  # reuse
    buf363 = empty((768, ), device='cpu', dtype=torch.float32)
    buf364 = empty((768, ), device='cpu', dtype=torch.float32)
    buf365 = reinterpret_tensor(buf343, (1, 512, 768), (393216, 768, 1), 0); del buf343  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_70(c_void_p(buf356.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(mul_52.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    del div_55
    del getitem_67
    del mul_52
    del primals_164
    buf366 = buf357; del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf365, (512, 768), (768, 1), 0), permute_491, out=buf366)
    del permute_491
    buf367 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf365, (768, 512), (1, 768), 0), view_246, out=buf367)
    del view_246
    buf368 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_71(c_void_p(buf365.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = reinterpret_tensor(buf342, (6, 512, 64), (32768, 64, 1), 0); del buf342  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_31, reinterpret_tensor(buf366, (6, 512, 64), (64, 768, 1), 0), out=buf369)
    del permute_default_31
    buf370 = reinterpret_tensor(buf356, (6, 512, 512), (262144, 512, 1), 0); del buf356  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf366, (6, 512, 64), (64, 768, 1), 0), permute_default_32, out=buf370)
    del permute_default_32
    buf371 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf372 = reinterpret_tensor(buf370, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf370  # reuse
    cpp_fused_72(c_void_p(buf372.data_ptr()), c_void_p(getitem_209.data_ptr()), c_void_p(alias_default_11.data_ptr()), c_void_p(buf371.data_ptr()))
    del alias_default_11
    del getitem_209
    buf373 = reinterpret_tensor(buf337, (6, 64, 512), (32768, 512, 1), 0); del buf337  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_33, reinterpret_tensor(buf372, (6, 512, 512), (262144, 512, 1), 0), out=buf373)
    del permute_default_33
    buf374 = reinterpret_tensor(buf333, (6, 512, 64), (32768, 64, 1), 0); del buf333  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf372, (6, 512, 512), (262144, 512, 1), 0), permute_default_34, out=buf374)
    del permute_default_34
    buf375 = reinterpret_tensor(buf303, (512, 384), (384, 1), 0); del buf303  # reuse
    cpp_fused_clone_73(c_void_p(buf366.data_ptr()), c_void_p(buf375.data_ptr()))
    buf376 = buf321; del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_501, reinterpret_tensor(buf375, (3072, 64, 1), (64, 1, 0), 0), out=buf376)
    del permute_501
    buf377 = buf311; del buf311  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf375, (3072, 64, 1), (64, 1, 0), 0), permute_502, out=buf377)
    del permute_502
    buf378 = buf312; del buf312  # reuse
    cpp_fused_col2im_74(c_void_p(buf378.data_ptr()))
    aten.index_put_(buf378, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf377, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf381 = reinterpret_tensor(buf375, (512, 384), (1, 512), 0); del buf375  # reuse
    buf384 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_75(c_void_p(buf378.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf384.data_ptr()))
    buf382 = buf366; del buf366  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf381, permute_506, out=buf382)
    del permute_506
    buf383 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (384, 512), (512, 1), 0), view_216, out=buf383)
    buf385 = reinterpret_tensor(buf371, (3072, 1, 1), (1, 3072, 3072), 0); del buf371  # reuse
    buf386 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf387 = buf376; del buf376  # reuse
    cpp_fused__softmax_backward_data_sum_76(c_void_p(buf387.data_ptr()), c_void_p(alias_37.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del alias_37
    buf388 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf387, (54, 512), (1, 54), 0), view_225, out=buf388)
    del view_225
    buf389 = reinterpret_tensor(buf381, (384, 512), (512, 1), 0); del buf381  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_123, reinterpret_tensor(buf387, (54, 512), (1, 54), 0), out=buf389)
    del permute_123
    buf390 = buf324; del buf324  # reuse
    buf408 = reinterpret_tensor(buf323, (512, 384), (1, 512), 0); del buf323  # reuse
    buf391 = reinterpret_tensor(buf308, (1, 384, 512), (196608, 512, 1), 0); del buf308  # reuse
    buf392 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_77(c_void_p(buf389.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()))
    del addmm_42
    del convolution_13
    del primals_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf393 = aten.convolution_backward(buf391, convolution_12, primals_157, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_12
    del primals_157
    buf394 = buf393[0]
    buf395 = buf393[1]
    del buf393
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf396 = aten.convolution_backward(buf394, permute_117, primals_156, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_117
    del primals_156
    buf397 = buf396[0]
    buf398 = buf396[1]
    del buf396
    buf399 = reinterpret_tensor(buf391, (512, 384), (384, 1), 0); del buf391  # reuse
    cpp_fused_view_78(c_void_p(buf369.data_ptr()), c_void_p(buf399.data_ptr()))
    buf400 = reinterpret_tensor(buf394, (512, 768), (768, 1), 0); del buf394  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf399, permute_520, out=buf400)
    del permute_520
    buf401 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (384, 512), (1, 384), 0), view_216, out=buf401)
    buf402 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf403 = reinterpret_tensor(buf373, (512, 384), (1, 512), 0); del buf373  # reuse
    cpp_fused_sum_view_79(c_void_p(buf403.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf402.data_ptr()))
    buf404 = reinterpret_tensor(buf365, (512, 768), (768, 1), 0); del buf365  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf403, permute_524, out=buf404)
    del permute_524
    buf405 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (384, 512), (512, 1), 0), view_216, out=buf405)
    buf406 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_80(c_void_p(buf403.data_ptr()), c_void_p(buf406.data_ptr()))
    buf409 = reinterpret_tensor(buf349, (512, 768), (768, 1), 0); del buf349  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf408, permute_528, out=buf409)
    del permute_528
    buf410 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (384, 512), (512, 1), 0), view_216, out=buf410)
    del view_216
    buf411 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf412 = buf362; del buf362  # reuse
    buf413 = buf361; del buf361  # reuse
    buf414 = buf360; del buf360  # reuse
    buf415 = reinterpret_tensor(buf338, (1, 512, 768), (393216, 768, 1), 0); del buf338  # reuse
    buf416 = empty((768, ), device='cpu', dtype=torch.float32)
    buf417 = empty((768, ), device='cpu', dtype=torch.float32)
    buf418 = reinterpret_tensor(buf334, (1, 512, 768), (393216, 768, 1), 0); del buf334  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_81(c_void_p(buf412.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    del buf382
    del buf397
    del div_57
    del getitem_61
    del mul_49
    del primals_148
    buf419 = reinterpret_tensor(buf372, (512, 3072), (3072, 1), 0); del buf372  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf418, (512, 768), (768, 1), 0), permute_532, out=buf419)
    del permute_532
    buf420 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf418, (768, 512), (1, 768), 0), view_214, out=buf420)
    del view_214
    buf421 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf422 = reinterpret_tensor(buf419, (1, 512, 3072), (1572864, 3072, 1), 0); del buf419  # reuse
    cpp_fused_gelu_gelu_backward_sum_82(c_void_p(buf422.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf421.data_ptr()))
    del addmm_40
    buf423 = reinterpret_tensor(buf418, (512, 768), (768, 1), 0); del buf418  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (512, 3072), (3072, 1), 0), permute_536, out=buf423)
    del permute_536
    buf424 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (3072, 512), (1, 3072), 0), view_212, out=buf424)
    del view_212
    buf425 = reinterpret_tensor(buf385, (1, 3072), (3072, 1), 0); del buf385  # reuse
    buf426 = buf414; del buf414  # reuse
    buf427 = buf413; del buf413  # reuse
    buf428 = buf412; del buf412  # reuse
    buf429 = empty((768, ), device='cpu', dtype=torch.float32)
    buf430 = empty((768, ), device='cpu', dtype=torch.float32)
    buf431 = reinterpret_tensor(buf409, (1, 512, 768), (393216, 768, 1), 0); del buf409  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83(c_void_p(buf422.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(mul_44.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()))
    del div_58
    del getitem_57
    del mul_44
    del primals_142
    buf432 = buf423; del buf423  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (512, 768), (768, 1), 0), permute_540, out=buf432)
    del permute_540
    buf433 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (768, 512), (1, 768), 0), view_210, out=buf433)
    del view_210
    buf434 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_84(c_void_p(buf431.data_ptr()), c_void_p(buf434.data_ptr()))
    buf435 = reinterpret_tensor(buf408, (6, 512, 64), (32768, 64, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_37, reinterpret_tensor(buf432, (6, 512, 64), (64, 768, 1), 0), out=buf435)
    del permute_default_37
    buf436 = reinterpret_tensor(buf422, (6, 512, 512), (262144, 512, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf432, (6, 512, 64), (64, 768, 1), 0), permute_default_38, out=buf436)
    del permute_default_38
    buf437 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf438 = reinterpret_tensor(buf436, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf436  # reuse
    cpp_fused_85(c_void_p(buf438.data_ptr()), c_void_p(getitem_211.data_ptr()), c_void_p(alias_default_13.data_ptr()), c_void_p(buf437.data_ptr()))
    del alias_default_13
    del getitem_211
    buf439 = reinterpret_tensor(buf403, (6, 64, 512), (32768, 512, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_39, reinterpret_tensor(buf438, (6, 512, 512), (262144, 512, 1), 0), out=buf439)
    del permute_default_39
    buf440 = reinterpret_tensor(buf399, (6, 512, 64), (32768, 64, 1), 0); del buf399  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf438, (6, 512, 512), (262144, 512, 1), 0), permute_default_40, out=buf440)
    del permute_default_40
    buf441 = reinterpret_tensor(buf369, (512, 384), (384, 1), 0); del buf369  # reuse
    cpp_fused_clone_86(c_void_p(buf432.data_ptr()), c_void_p(buf441.data_ptr()))
    buf442 = buf387; del buf387  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_550, reinterpret_tensor(buf441, (3072, 64, 1), (64, 1, 0), 0), out=buf442)
    del permute_550
    buf443 = buf377; del buf377  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf441, (3072, 64, 1), (64, 1, 0), 0), permute_551, out=buf443)
    del permute_551
    buf444 = buf378; del buf378  # reuse
    cpp_fused_col2im_87(c_void_p(buf444.data_ptr()))
    aten.index_put_(buf444, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf443, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf447 = reinterpret_tensor(buf441, (512, 384), (1, 512), 0); del buf441  # reuse
    buf450 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_88(c_void_p(buf444.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf450.data_ptr()))
    buf448 = buf432; del buf432  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf447, permute_555, out=buf448)
    del permute_555
    buf449 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf447, (384, 512), (512, 1), 0), view_180, out=buf449)
    buf451 = reinterpret_tensor(buf437, (3072, 1, 1), (1, 3072, 3072), 0); del buf437  # reuse
    buf452 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf453 = buf442; del buf442  # reuse
    cpp_fused__softmax_backward_data_sum_89(c_void_p(buf453.data_ptr()), c_void_p(alias_39.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()))
    del alias_39
    buf454 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (54, 512), (1, 54), 0), view_189, out=buf454)
    del view_189
    buf455 = reinterpret_tensor(buf447, (384, 512), (512, 1), 0); del buf447  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_104, reinterpret_tensor(buf453, (54, 512), (1, 54), 0), out=buf455)
    del permute_104
    buf456 = buf390; del buf390  # reuse
    buf474 = reinterpret_tensor(buf389, (512, 384), (1, 512), 0); del buf389  # reuse
    buf457 = reinterpret_tensor(buf374, (1, 384, 512), (196608, 512, 1), 0); del buf374  # reuse
    buf458 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_90(c_void_p(buf455.data_ptr()), c_void_p(addmm_35.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    del addmm_35
    del convolution_11
    del primals_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf459 = aten.convolution_backward(buf457, convolution_10, primals_135, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_10
    del primals_135
    buf460 = buf459[0]
    buf461 = buf459[1]
    del buf459
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf462 = aten.convolution_backward(buf460, permute_98, primals_134, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_98
    del primals_134
    buf463 = buf462[0]
    buf464 = buf462[1]
    del buf462
    buf465 = reinterpret_tensor(buf457, (512, 384), (384, 1), 0); del buf457  # reuse
    cpp_fused_view_91(c_void_p(buf435.data_ptr()), c_void_p(buf465.data_ptr()))
    buf466 = reinterpret_tensor(buf460, (512, 768), (768, 1), 0); del buf460  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf465, permute_569, out=buf466)
    del permute_569
    buf467 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf465, (384, 512), (1, 384), 0), view_180, out=buf467)
    buf468 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf469 = reinterpret_tensor(buf439, (512, 384), (1, 512), 0); del buf439  # reuse
    cpp_fused_sum_view_92(c_void_p(buf469.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf468.data_ptr()))
    buf470 = reinterpret_tensor(buf431, (512, 768), (768, 1), 0); del buf431  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf469, permute_573, out=buf470)
    del permute_573
    buf471 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf469, (384, 512), (512, 1), 0), view_180, out=buf471)
    buf472 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_93(c_void_p(buf469.data_ptr()), c_void_p(buf472.data_ptr()))
    buf475 = reinterpret_tensor(buf415, (512, 768), (768, 1), 0); del buf415  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf474, permute_577, out=buf475)
    del permute_577
    buf476 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (384, 512), (512, 1), 0), view_180, out=buf476)
    del view_180
    buf477 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf478 = buf428; del buf428  # reuse
    buf479 = buf427; del buf427  # reuse
    buf480 = buf426; del buf426  # reuse
    buf481 = reinterpret_tensor(buf404, (1, 512, 768), (393216, 768, 1), 0); del buf404  # reuse
    buf482 = empty((768, ), device='cpu', dtype=torch.float32)
    buf483 = empty((768, ), device='cpu', dtype=torch.float32)
    buf484 = reinterpret_tensor(buf400, (1, 512, 768), (393216, 768, 1), 0); del buf400  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_94(c_void_p(buf478.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(mul_41.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()))
    del buf448
    del buf463
    del div_60
    del getitem_51
    del mul_41
    del primals_126
    buf485 = reinterpret_tensor(buf438, (512, 3072), (3072, 1), 0); del buf438  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf484, (512, 768), (768, 1), 0), permute_581, out=buf485)
    del permute_581
    buf486 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf484, (768, 512), (1, 768), 0), view_178, out=buf486)
    del view_178
    buf487 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf488 = reinterpret_tensor(buf485, (1, 512, 3072), (1572864, 3072, 1), 0); del buf485  # reuse
    cpp_fused_gelu_gelu_backward_sum_95(c_void_p(buf488.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(addmm_33.data_ptr()), c_void_p(buf487.data_ptr()))
    del addmm_33
    buf489 = reinterpret_tensor(buf484, (512, 768), (768, 1), 0); del buf484  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf488, (512, 3072), (3072, 1), 0), permute_585, out=buf489)
    del permute_585
    buf490 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf488, (3072, 512), (1, 3072), 0), view_176, out=buf490)
    del view_176
    buf491 = reinterpret_tensor(buf451, (1, 3072), (3072, 1), 0); del buf451  # reuse
    buf492 = buf480; del buf480  # reuse
    buf493 = buf479; del buf479  # reuse
    buf494 = buf478; del buf478  # reuse
    buf495 = empty((768, ), device='cpu', dtype=torch.float32)
    buf496 = empty((768, ), device='cpu', dtype=torch.float32)
    buf497 = reinterpret_tensor(buf475, (1, 512, 768), (393216, 768, 1), 0); del buf475  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_96(c_void_p(buf488.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(mul_36.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()))
    del div_61
    del getitem_47
    del mul_36
    del primals_120
    buf498 = buf489; del buf489  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf497, (512, 768), (768, 1), 0), permute_589, out=buf498)
    del permute_589
    buf499 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf497, (768, 512), (1, 768), 0), view_174, out=buf499)
    del view_174
    buf500 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_97(c_void_p(buf497.data_ptr()), c_void_p(buf500.data_ptr()))
    buf501 = reinterpret_tensor(buf474, (6, 512, 64), (32768, 64, 1), 0); del buf474  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_43, reinterpret_tensor(buf498, (6, 512, 64), (64, 768, 1), 0), out=buf501)
    del permute_default_43
    buf502 = reinterpret_tensor(buf488, (6, 512, 512), (262144, 512, 1), 0); del buf488  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf498, (6, 512, 64), (64, 768, 1), 0), permute_default_44, out=buf502)
    del permute_default_44
    buf503 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf504 = reinterpret_tensor(buf502, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf502  # reuse
    cpp_fused_98(c_void_p(buf504.data_ptr()), c_void_p(getitem_213.data_ptr()), c_void_p(alias_default_15.data_ptr()), c_void_p(buf503.data_ptr()))
    del alias_default_15
    del getitem_213
    buf505 = reinterpret_tensor(buf469, (6, 64, 512), (32768, 512, 1), 0); del buf469  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_45, reinterpret_tensor(buf504, (6, 512, 512), (262144, 512, 1), 0), out=buf505)
    del permute_default_45
    buf506 = reinterpret_tensor(buf465, (6, 512, 64), (32768, 64, 1), 0); del buf465  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf504, (6, 512, 512), (262144, 512, 1), 0), permute_default_46, out=buf506)
    del permute_default_46
    buf507 = reinterpret_tensor(buf435, (512, 384), (384, 1), 0); del buf435  # reuse
    cpp_fused_clone_99(c_void_p(buf498.data_ptr()), c_void_p(buf507.data_ptr()))
    buf508 = buf453; del buf453  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_599, reinterpret_tensor(buf507, (3072, 64, 1), (64, 1, 0), 0), out=buf508)
    del permute_599
    buf509 = buf443; del buf443  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf507, (3072, 64, 1), (64, 1, 0), 0), permute_600, out=buf509)
    del permute_600
    buf510 = buf444; del buf444  # reuse
    cpp_fused_col2im_100(c_void_p(buf510.data_ptr()))
    aten.index_put_(buf510, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf509, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf513 = reinterpret_tensor(buf507, (512, 384), (1, 512), 0); del buf507  # reuse
    buf516 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_101(c_void_p(buf510.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf516.data_ptr()))
    buf514 = buf498; del buf498  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf513, permute_604, out=buf514)
    del permute_604
    buf515 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf513, (384, 512), (512, 1), 0), view_144, out=buf515)
    buf517 = reinterpret_tensor(buf503, (3072, 1, 1), (1, 3072, 3072), 0); del buf503  # reuse
    buf518 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf519 = buf508; del buf508  # reuse
    cpp_fused__softmax_backward_data_sum_102(c_void_p(buf519.data_ptr()), c_void_p(alias_41.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    del alias_41
    buf520 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf519, (54, 512), (1, 54), 0), view_153, out=buf520)
    del view_153
    buf521 = reinterpret_tensor(buf513, (384, 512), (512, 1), 0); del buf513  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_85, reinterpret_tensor(buf519, (54, 512), (1, 54), 0), out=buf521)
    del permute_85
    buf522 = buf456; del buf456  # reuse
    buf540 = reinterpret_tensor(buf455, (512, 384), (1, 512), 0); del buf455  # reuse
    buf523 = reinterpret_tensor(buf440, (1, 384, 512), (196608, 512, 1), 0); del buf440  # reuse
    buf524 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_103(c_void_p(buf521.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()))
    del addmm_28
    del convolution_9
    del primals_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf525 = aten.convolution_backward(buf523, convolution_8, primals_113, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_8
    del primals_113
    buf526 = buf525[0]
    buf527 = buf525[1]
    del buf525
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf528 = aten.convolution_backward(buf526, permute_79, primals_112, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_79
    del primals_112
    buf529 = buf528[0]
    buf530 = buf528[1]
    del buf528
    buf531 = reinterpret_tensor(buf523, (512, 384), (384, 1), 0); del buf523  # reuse
    cpp_fused_view_104(c_void_p(buf501.data_ptr()), c_void_p(buf531.data_ptr()))
    buf532 = reinterpret_tensor(buf526, (512, 768), (768, 1), 0); del buf526  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf531, permute_618, out=buf532)
    del permute_618
    buf533 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf531, (384, 512), (1, 384), 0), view_144, out=buf533)
    buf534 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf535 = reinterpret_tensor(buf505, (512, 384), (1, 512), 0); del buf505  # reuse
    cpp_fused_sum_view_105(c_void_p(buf535.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf534.data_ptr()))
    buf536 = reinterpret_tensor(buf497, (512, 768), (768, 1), 0); del buf497  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf535, permute_622, out=buf536)
    del permute_622
    buf537 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf535, (384, 512), (512, 1), 0), view_144, out=buf537)
    buf538 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_106(c_void_p(buf535.data_ptr()), c_void_p(buf538.data_ptr()))
    buf541 = reinterpret_tensor(buf481, (512, 768), (768, 1), 0); del buf481  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf540, permute_626, out=buf541)
    del permute_626
    buf542 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf540, (384, 512), (512, 1), 0), view_144, out=buf542)
    del view_144
    buf543 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf544 = buf494; del buf494  # reuse
    buf545 = buf493; del buf493  # reuse
    buf546 = buf492; del buf492  # reuse
    buf547 = reinterpret_tensor(buf470, (1, 512, 768), (393216, 768, 1), 0); del buf470  # reuse
    buf548 = empty((768, ), device='cpu', dtype=torch.float32)
    buf549 = empty((768, ), device='cpu', dtype=torch.float32)
    buf550 = reinterpret_tensor(buf466, (1, 512, 768), (393216, 768, 1), 0); del buf466  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_107(c_void_p(buf544.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(mul_33.data_ptr()), c_void_p(div_63.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()))
    del buf514
    del buf529
    del div_63
    del getitem_41
    del mul_33
    del primals_104
    buf551 = reinterpret_tensor(buf504, (512, 3072), (3072, 1), 0); del buf504  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf550, (512, 768), (768, 1), 0), permute_630, out=buf551)
    del permute_630
    buf552 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf550, (768, 512), (1, 768), 0), view_142, out=buf552)
    del view_142
    buf553 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf554 = reinterpret_tensor(buf551, (1, 512, 3072), (1572864, 3072, 1), 0); del buf551  # reuse
    cpp_fused_gelu_gelu_backward_sum_108(c_void_p(buf554.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf553.data_ptr()))
    del addmm_26
    buf555 = reinterpret_tensor(buf550, (512, 768), (768, 1), 0); del buf550  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf554, (512, 3072), (3072, 1), 0), permute_634, out=buf555)
    del permute_634
    buf556 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf554, (3072, 512), (1, 3072), 0), view_140, out=buf556)
    del view_140
    buf557 = reinterpret_tensor(buf517, (1, 3072), (3072, 1), 0); del buf517  # reuse
    buf558 = buf546; del buf546  # reuse
    buf559 = buf545; del buf545  # reuse
    buf560 = buf544; del buf544  # reuse
    buf561 = empty((768, ), device='cpu', dtype=torch.float32)
    buf562 = empty((768, ), device='cpu', dtype=torch.float32)
    buf563 = reinterpret_tensor(buf541, (1, 512, 768), (393216, 768, 1), 0); del buf541  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_109(c_void_p(buf554.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(div_64.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()))
    del div_64
    del getitem_37
    del mul_28
    del primals_98
    buf564 = buf555; del buf555  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf563, (512, 768), (768, 1), 0), permute_638, out=buf564)
    del permute_638
    buf565 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf563, (768, 512), (1, 768), 0), view_138, out=buf565)
    del view_138
    buf566 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_110(c_void_p(buf563.data_ptr()), c_void_p(buf566.data_ptr()))
    buf567 = reinterpret_tensor(buf540, (6, 512, 64), (32768, 64, 1), 0); del buf540  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_49, reinterpret_tensor(buf564, (6, 512, 64), (64, 768, 1), 0), out=buf567)
    del permute_default_49
    buf568 = reinterpret_tensor(buf554, (6, 512, 512), (262144, 512, 1), 0); del buf554  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf564, (6, 512, 64), (64, 768, 1), 0), permute_default_50, out=buf568)
    del permute_default_50
    buf569 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf570 = reinterpret_tensor(buf568, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf568  # reuse
    cpp_fused_111(c_void_p(buf570.data_ptr()), c_void_p(getitem_215.data_ptr()), c_void_p(alias_default_17.data_ptr()), c_void_p(buf569.data_ptr()))
    del alias_default_17
    del getitem_215
    buf571 = reinterpret_tensor(buf535, (6, 64, 512), (32768, 512, 1), 0); del buf535  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_51, reinterpret_tensor(buf570, (6, 512, 512), (262144, 512, 1), 0), out=buf571)
    del permute_default_51
    buf572 = reinterpret_tensor(buf531, (6, 512, 64), (32768, 64, 1), 0); del buf531  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf570, (6, 512, 512), (262144, 512, 1), 0), permute_default_52, out=buf572)
    del permute_default_52
    buf573 = reinterpret_tensor(buf501, (512, 384), (384, 1), 0); del buf501  # reuse
    cpp_fused_clone_112(c_void_p(buf564.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = buf519; del buf519  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_648, reinterpret_tensor(buf573, (3072, 64, 1), (64, 1, 0), 0), out=buf574)
    del permute_648
    buf575 = buf509; del buf509  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf573, (3072, 64, 1), (64, 1, 0), 0), permute_649, out=buf575)
    del permute_649
    buf576 = buf510; del buf510  # reuse
    cpp_fused_col2im_113(c_void_p(buf576.data_ptr()))
    aten.index_put_(buf576, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf575, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf579 = reinterpret_tensor(buf573, (512, 384), (1, 512), 0); del buf573  # reuse
    buf582 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_114(c_void_p(buf576.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf582.data_ptr()))
    buf580 = buf564; del buf564  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf579, permute_653, out=buf580)
    del permute_653
    buf581 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf579, (384, 512), (512, 1), 0), view_108, out=buf581)
    buf583 = reinterpret_tensor(buf569, (3072, 1, 1), (1, 3072, 3072), 0); del buf569  # reuse
    buf584 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf585 = buf574; del buf574  # reuse
    cpp_fused__softmax_backward_data_sum_115(c_void_p(buf585.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()))
    del alias_43
    buf586 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf585, (54, 512), (1, 54), 0), view_117, out=buf586)
    del view_117
    buf587 = reinterpret_tensor(buf579, (384, 512), (512, 1), 0); del buf579  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_66, reinterpret_tensor(buf585, (54, 512), (1, 54), 0), out=buf587)
    del permute_66
    buf588 = buf522; del buf522  # reuse
    buf606 = reinterpret_tensor(buf521, (512, 384), (1, 512), 0); del buf521  # reuse
    buf589 = reinterpret_tensor(buf506, (1, 384, 512), (196608, 512, 1), 0); del buf506  # reuse
    buf590 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_116(c_void_p(buf587.data_ptr()), c_void_p(addmm_21.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf590.data_ptr()))
    del addmm_21
    del convolution_7
    del primals_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf591 = aten.convolution_backward(buf589, convolution_6, primals_91, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_6
    del primals_91
    buf592 = buf591[0]
    buf593 = buf591[1]
    del buf591
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf594 = aten.convolution_backward(buf592, permute_60, primals_90, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_60
    del primals_90
    buf595 = buf594[0]
    buf596 = buf594[1]
    del buf594
    buf597 = reinterpret_tensor(buf589, (512, 384), (384, 1), 0); del buf589  # reuse
    cpp_fused_view_117(c_void_p(buf567.data_ptr()), c_void_p(buf597.data_ptr()))
    buf598 = reinterpret_tensor(buf592, (512, 768), (768, 1), 0); del buf592  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf597, permute_667, out=buf598)
    del permute_667
    buf599 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf597, (384, 512), (1, 384), 0), view_108, out=buf599)
    buf600 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf601 = reinterpret_tensor(buf571, (512, 384), (1, 512), 0); del buf571  # reuse
    cpp_fused_sum_view_118(c_void_p(buf601.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf600.data_ptr()))
    buf602 = reinterpret_tensor(buf563, (512, 768), (768, 1), 0); del buf563  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf601, permute_671, out=buf602)
    del permute_671
    buf603 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf601, (384, 512), (512, 1), 0), view_108, out=buf603)
    buf604 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_119(c_void_p(buf601.data_ptr()), c_void_p(buf604.data_ptr()))
    buf607 = reinterpret_tensor(buf547, (512, 768), (768, 1), 0); del buf547  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf606, permute_675, out=buf607)
    del permute_675
    buf608 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf606, (384, 512), (512, 1), 0), view_108, out=buf608)
    del view_108
    buf609 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf610 = buf560; del buf560  # reuse
    buf611 = buf559; del buf559  # reuse
    buf612 = buf558; del buf558  # reuse
    buf613 = reinterpret_tensor(buf536, (1, 512, 768), (393216, 768, 1), 0); del buf536  # reuse
    buf614 = empty((768, ), device='cpu', dtype=torch.float32)
    buf615 = empty((768, ), device='cpu', dtype=torch.float32)
    buf616 = reinterpret_tensor(buf532, (1, 512, 768), (393216, 768, 1), 0); del buf532  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_120(c_void_p(buf610.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(mul_25.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf616.data_ptr()))
    del buf580
    del buf595
    del div_66
    del getitem_31
    del mul_25
    del primals_82
    buf617 = reinterpret_tensor(buf570, (512, 3072), (3072, 1), 0); del buf570  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf616, (512, 768), (768, 1), 0), permute_679, out=buf617)
    del permute_679
    buf618 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf616, (768, 512), (1, 768), 0), view_106, out=buf618)
    del view_106
    buf619 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf620 = reinterpret_tensor(buf617, (1, 512, 3072), (1572864, 3072, 1), 0); del buf617  # reuse
    cpp_fused_gelu_gelu_backward_sum_121(c_void_p(buf620.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(buf619.data_ptr()))
    del addmm_19
    buf621 = reinterpret_tensor(buf616, (512, 768), (768, 1), 0); del buf616  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf620, (512, 3072), (3072, 1), 0), permute_683, out=buf621)
    del permute_683
    buf622 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf620, (3072, 512), (1, 3072), 0), view_104, out=buf622)
    del view_104
    buf623 = reinterpret_tensor(buf583, (1, 3072), (3072, 1), 0); del buf583  # reuse
    buf624 = buf612; del buf612  # reuse
    buf625 = buf611; del buf611  # reuse
    buf626 = buf610; del buf610  # reuse
    buf627 = empty((768, ), device='cpu', dtype=torch.float32)
    buf628 = empty((768, ), device='cpu', dtype=torch.float32)
    buf629 = reinterpret_tensor(buf607, (1, 512, 768), (393216, 768, 1), 0); del buf607  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_122(c_void_p(buf620.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(mul_20.data_ptr()), c_void_p(div_67.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()))
    del div_67
    del getitem_27
    del mul_20
    del primals_76
    buf630 = buf621; del buf621  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (512, 768), (768, 1), 0), permute_687, out=buf630)
    del permute_687
    buf631 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (768, 512), (1, 768), 0), view_102, out=buf631)
    del view_102
    buf632 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_123(c_void_p(buf629.data_ptr()), c_void_p(buf632.data_ptr()))
    buf633 = reinterpret_tensor(buf606, (6, 512, 64), (32768, 64, 1), 0); del buf606  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_55, reinterpret_tensor(buf630, (6, 512, 64), (64, 768, 1), 0), out=buf633)
    del permute_default_55
    buf634 = reinterpret_tensor(buf620, (6, 512, 512), (262144, 512, 1), 0); del buf620  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf630, (6, 512, 64), (64, 768, 1), 0), permute_default_56, out=buf634)
    del permute_default_56
    buf635 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf636 = reinterpret_tensor(buf634, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf634  # reuse
    cpp_fused_124(c_void_p(buf636.data_ptr()), c_void_p(getitem_217.data_ptr()), c_void_p(alias_default_19.data_ptr()), c_void_p(buf635.data_ptr()))
    del alias_default_19
    del getitem_217
    buf637 = reinterpret_tensor(buf601, (6, 64, 512), (32768, 512, 1), 0); del buf601  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_57, reinterpret_tensor(buf636, (6, 512, 512), (262144, 512, 1), 0), out=buf637)
    del permute_default_57
    buf638 = reinterpret_tensor(buf597, (6, 512, 64), (32768, 64, 1), 0); del buf597  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf636, (6, 512, 512), (262144, 512, 1), 0), permute_default_58, out=buf638)
    del permute_default_58
    buf639 = reinterpret_tensor(buf567, (512, 384), (384, 1), 0); del buf567  # reuse
    cpp_fused_clone_125(c_void_p(buf630.data_ptr()), c_void_p(buf639.data_ptr()))
    buf640 = buf585; del buf585  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_697, reinterpret_tensor(buf639, (3072, 64, 1), (64, 1, 0), 0), out=buf640)
    del permute_697
    buf641 = buf575; del buf575  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf639, (3072, 64, 1), (64, 1, 0), 0), permute_698, out=buf641)
    del permute_698
    buf642 = buf576; del buf576  # reuse
    cpp_fused_col2im_126(c_void_p(buf642.data_ptr()))
    aten.index_put_(buf642, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf641, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf645 = reinterpret_tensor(buf639, (512, 384), (1, 512), 0); del buf639  # reuse
    buf648 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_127(c_void_p(buf642.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf648.data_ptr()))
    buf646 = buf630; del buf630  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf645, permute_702, out=buf646)
    del permute_702
    buf647 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf645, (384, 512), (512, 1), 0), view_72, out=buf647)
    buf649 = reinterpret_tensor(buf635, (3072, 1, 1), (1, 3072, 3072), 0); del buf635  # reuse
    buf650 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf651 = buf640; del buf640  # reuse
    cpp_fused__softmax_backward_data_sum_128(c_void_p(buf651.data_ptr()), c_void_p(alias_45.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()))
    del alias_45
    buf652 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf651, (54, 512), (1, 54), 0), view_81, out=buf652)
    del view_81
    buf653 = reinterpret_tensor(buf645, (384, 512), (512, 1), 0); del buf645  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_47, reinterpret_tensor(buf651, (54, 512), (1, 54), 0), out=buf653)
    del permute_47
    buf654 = buf588; del buf588  # reuse
    buf672 = reinterpret_tensor(buf587, (512, 384), (1, 512), 0); del buf587  # reuse
    buf655 = reinterpret_tensor(buf572, (1, 384, 512), (196608, 512, 1), 0); del buf572  # reuse
    buf656 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_129(c_void_p(buf653.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()))
    del addmm_14
    del convolution_5
    del primals_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf657 = aten.convolution_backward(buf655, convolution_4, primals_69, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_4
    del primals_69
    buf658 = buf657[0]
    buf659 = buf657[1]
    del buf657
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf660 = aten.convolution_backward(buf658, permute_41, primals_68, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_41
    del primals_68
    buf661 = buf660[0]
    buf662 = buf660[1]
    del buf660
    buf663 = reinterpret_tensor(buf655, (512, 384), (384, 1), 0); del buf655  # reuse
    cpp_fused_view_130(c_void_p(buf633.data_ptr()), c_void_p(buf663.data_ptr()))
    buf664 = reinterpret_tensor(buf658, (512, 768), (768, 1), 0); del buf658  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf663, permute_716, out=buf664)
    del permute_716
    buf665 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf663, (384, 512), (1, 384), 0), view_72, out=buf665)
    buf666 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf667 = reinterpret_tensor(buf637, (512, 384), (1, 512), 0); del buf637  # reuse
    cpp_fused_sum_view_131(c_void_p(buf667.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf666.data_ptr()))
    buf668 = reinterpret_tensor(buf629, (512, 768), (768, 1), 0); del buf629  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf667, permute_720, out=buf668)
    del permute_720
    buf669 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf667, (384, 512), (512, 1), 0), view_72, out=buf669)
    buf670 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_132(c_void_p(buf667.data_ptr()), c_void_p(buf670.data_ptr()))
    buf673 = reinterpret_tensor(buf613, (512, 768), (768, 1), 0); del buf613  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf672, permute_724, out=buf673)
    del permute_724
    buf674 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf672, (384, 512), (512, 1), 0), view_72, out=buf674)
    del view_72
    buf675 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf676 = buf626; del buf626  # reuse
    buf677 = buf625; del buf625  # reuse
    buf678 = buf624; del buf624  # reuse
    buf679 = reinterpret_tensor(buf602, (1, 512, 768), (393216, 768, 1), 0); del buf602  # reuse
    buf680 = empty((768, ), device='cpu', dtype=torch.float32)
    buf681 = empty((768, ), device='cpu', dtype=torch.float32)
    buf682 = reinterpret_tensor(buf598, (1, 512, 768), (393216, 768, 1), 0); del buf598  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_133(c_void_p(buf676.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(div_69.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()))
    del buf646
    del buf661
    del div_69
    del getitem_21
    del mul_17
    del primals_60
    buf683 = reinterpret_tensor(buf636, (512, 3072), (3072, 1), 0); del buf636  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf682, (512, 768), (768, 1), 0), permute_728, out=buf683)
    del permute_728
    buf684 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf682, (768, 512), (1, 768), 0), view_70, out=buf684)
    del view_70
    buf685 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf686 = reinterpret_tensor(buf683, (1, 512, 3072), (1572864, 3072, 1), 0); del buf683  # reuse
    cpp_fused_gelu_gelu_backward_sum_134(c_void_p(buf686.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(addmm_12.data_ptr()), c_void_p(buf685.data_ptr()))
    del addmm_12
    buf687 = reinterpret_tensor(buf682, (512, 768), (768, 1), 0); del buf682  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf686, (512, 3072), (3072, 1), 0), permute_732, out=buf687)
    del permute_732
    buf688 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf686, (3072, 512), (1, 3072), 0), view_68, out=buf688)
    del view_68
    buf689 = reinterpret_tensor(buf649, (1, 3072), (3072, 1), 0); del buf649  # reuse
    buf690 = buf678; del buf678  # reuse
    buf691 = buf677; del buf677  # reuse
    buf692 = buf676; del buf676  # reuse
    buf693 = empty((768, ), device='cpu', dtype=torch.float32)
    buf694 = empty((768, ), device='cpu', dtype=torch.float32)
    buf695 = reinterpret_tensor(buf673, (1, 512, 768), (393216, 768, 1), 0); del buf673  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_135(c_void_p(buf686.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(mul_12.data_ptr()), c_void_p(div_70.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf692.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()))
    del div_70
    del getitem_17
    del mul_12
    del primals_54
    buf696 = buf687; del buf687  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf695, (512, 768), (768, 1), 0), permute_736, out=buf696)
    del permute_736
    buf697 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf695, (768, 512), (1, 768), 0), view_66, out=buf697)
    del view_66
    buf698 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_136(c_void_p(buf695.data_ptr()), c_void_p(buf698.data_ptr()))
    buf699 = reinterpret_tensor(buf672, (6, 512, 64), (32768, 64, 1), 0); del buf672  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_61, reinterpret_tensor(buf696, (6, 512, 64), (64, 768, 1), 0), out=buf699)
    del permute_default_61
    buf700 = reinterpret_tensor(buf686, (6, 512, 512), (262144, 512, 1), 0); del buf686  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf696, (6, 512, 64), (64, 768, 1), 0), permute_default_62, out=buf700)
    del permute_default_62
    buf701 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf702 = reinterpret_tensor(buf700, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf700  # reuse
    cpp_fused_137(c_void_p(buf702.data_ptr()), c_void_p(getitem_219.data_ptr()), c_void_p(alias_default_21.data_ptr()), c_void_p(buf701.data_ptr()))
    del alias_default_21
    del getitem_219
    buf703 = reinterpret_tensor(buf667, (6, 64, 512), (32768, 512, 1), 0); del buf667  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_63, reinterpret_tensor(buf702, (6, 512, 512), (262144, 512, 1), 0), out=buf703)
    del permute_default_63
    buf704 = reinterpret_tensor(buf663, (6, 512, 64), (32768, 64, 1), 0); del buf663  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf702, (6, 512, 512), (262144, 512, 1), 0), permute_default_64, out=buf704)
    del permute_default_64
    buf705 = reinterpret_tensor(buf633, (512, 384), (384, 1), 0); del buf633  # reuse
    cpp_fused_clone_138(c_void_p(buf696.data_ptr()), c_void_p(buf705.data_ptr()))
    buf706 = buf651; del buf651  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_746, reinterpret_tensor(buf705, (3072, 64, 1), (64, 1, 0), 0), out=buf706)
    del permute_746
    buf707 = buf641; del buf641  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf705, (3072, 64, 1), (64, 1, 0), 0), permute_747, out=buf707)
    del permute_747
    buf708 = buf642; del buf642  # reuse
    cpp_fused_col2im_139(c_void_p(buf708.data_ptr()))
    aten.index_put_(buf708, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf707, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    buf711 = reinterpret_tensor(buf705, (512, 384), (1, 512), 0); del buf705  # reuse
    buf714 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_140(c_void_p(buf708.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf714.data_ptr()))
    buf712 = buf696; del buf696  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf711, permute_751, out=buf712)
    del permute_751
    buf713 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf711, (384, 512), (512, 1), 0), view_36, out=buf713)
    buf715 = reinterpret_tensor(buf701, (3072, 1, 1), (1, 3072, 3072), 0); del buf701  # reuse
    buf716 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf717 = buf706; del buf706  # reuse
    cpp_fused__softmax_backward_data_sum_141(c_void_p(buf717.data_ptr()), c_void_p(alias_47.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()))
    del alias_47
    buf718 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf717, (54, 512), (1, 54), 0), view_45, out=buf718)
    del view_45
    buf719 = reinterpret_tensor(buf711, (384, 512), (512, 1), 0); del buf711  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_28, reinterpret_tensor(buf717, (54, 512), (1, 54), 0), out=buf719)
    del permute_28
    buf720 = buf654; del buf654  # reuse
    buf738 = reinterpret_tensor(buf653, (512, 384), (1, 512), 0); del buf653  # reuse
    buf721 = reinterpret_tensor(buf638, (1, 384, 512), (196608, 512, 1), 0); del buf638  # reuse
    buf722 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_142(c_void_p(buf719.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()))
    del addmm_7
    del convolution_3
    del primals_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf723 = aten.convolution_backward(buf721, convolution_2, primals_47, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution_2
    del primals_47
    buf724 = buf723[0]
    buf725 = buf723[1]
    del buf723
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf726 = aten.convolution_backward(buf724, permute_22, primals_46, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_22
    del primals_46
    buf727 = buf726[0]
    buf728 = buf726[1]
    del buf726
    buf729 = reinterpret_tensor(buf721, (512, 384), (384, 1), 0); del buf721  # reuse
    cpp_fused_view_143(c_void_p(buf699.data_ptr()), c_void_p(buf729.data_ptr()))
    buf730 = reinterpret_tensor(buf724, (512, 768), (768, 1), 0); del buf724  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf729, permute_765, out=buf730)
    del permute_765
    buf731 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf729, (384, 512), (1, 384), 0), view_36, out=buf731)
    buf732 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf733 = reinterpret_tensor(buf703, (512, 384), (1, 512), 0); del buf703  # reuse
    cpp_fused_sum_view_144(c_void_p(buf733.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf732.data_ptr()))
    buf734 = reinterpret_tensor(buf695, (512, 768), (768, 1), 0); del buf695  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf733, permute_769, out=buf734)
    del permute_769
    buf735 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf733, (384, 512), (512, 1), 0), view_36, out=buf735)
    buf736 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_145(c_void_p(buf733.data_ptr()), c_void_p(buf736.data_ptr()))
    buf739 = reinterpret_tensor(buf679, (512, 768), (768, 1), 0); del buf679  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf738, permute_773, out=buf739)
    del permute_773
    buf740 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf738, (384, 512), (512, 1), 0), view_36, out=buf740)
    del view_36
    buf741 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf742 = buf692; del buf692  # reuse
    buf743 = buf691; del buf691  # reuse
    buf744 = buf690; del buf690  # reuse
    buf745 = reinterpret_tensor(buf668, (1, 512, 768), (393216, 768, 1), 0); del buf668  # reuse
    buf746 = empty((768, ), device='cpu', dtype=torch.float32)
    buf747 = empty((768, ), device='cpu', dtype=torch.float32)
    buf748 = reinterpret_tensor(buf664, (1, 512, 768), (393216, 768, 1), 0); del buf664  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_146(c_void_p(buf742.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_72.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf748.data_ptr()))
    del buf712
    del div_72
    del getitem_11
    del mul_9
    del primals_38
    buf749 = reinterpret_tensor(buf702, (512, 3072), (3072, 1), 0); del buf702  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf748, (512, 768), (768, 1), 0), permute_777, out=buf749)
    del permute_777
    buf750 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf748, (768, 512), (1, 768), 0), view_34, out=buf750)
    del view_34
    buf751 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf752 = reinterpret_tensor(buf749, (1, 512, 3072), (1572864, 3072, 1), 0); del buf749  # reuse
    cpp_fused_gelu_gelu_backward_sum_147(c_void_p(buf752.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(buf751.data_ptr()))
    del addmm_5
    buf753 = reinterpret_tensor(buf748, (512, 768), (768, 1), 0); del buf748  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf752, (512, 3072), (3072, 1), 0), permute_781, out=buf753)
    del permute_781
    buf754 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf752, (3072, 512), (1, 3072), 0), view_32, out=buf754)
    del view_32
    buf755 = reinterpret_tensor(buf715, (1, 3072), (3072, 1), 0); del buf715  # reuse
    buf756 = buf744; del buf744  # reuse
    buf757 = buf743; del buf743  # reuse
    buf758 = buf742; del buf742  # reuse
    buf759 = empty((768, ), device='cpu', dtype=torch.float32)
    buf760 = empty((768, ), device='cpu', dtype=torch.float32)
    buf761 = reinterpret_tensor(buf739, (1, 512, 768), (393216, 768, 1), 0); del buf739  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_148(c_void_p(buf752.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(mul_4.data_ptr()), c_void_p(div_73.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf759.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf761.data_ptr()))
    del div_73
    del getitem_7
    del mul_4
    del primals_32
    buf762 = buf753; del buf753  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf761, (512, 768), (768, 1), 0), permute_785, out=buf762)
    del permute_785
    buf763 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf761, (768, 512), (1, 768), 0), view_30, out=buf763)
    del view_30
    buf764 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_149(c_void_p(buf761.data_ptr()), c_void_p(buf764.data_ptr()))
    buf765 = reinterpret_tensor(buf738, (6, 512, 64), (32768, 64, 1), 0); del buf738  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_67, reinterpret_tensor(buf762, (6, 512, 64), (64, 768, 1), 0), out=buf765)
    del permute_default_67
    buf766 = reinterpret_tensor(buf752, (6, 512, 512), (262144, 512, 1), 0); del buf752  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf762, (6, 512, 64), (64, 768, 1), 0), permute_default_68, out=buf766)
    del permute_default_68
    buf767 = empty_strided((1, 6, 512, 1), (3072, 1, 6, 3072), device='cpu', dtype=torch.float32)
    buf768 = reinterpret_tensor(buf766, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf766  # reuse
    cpp_fused_150(c_void_p(buf768.data_ptr()), c_void_p(getitem_221.data_ptr()), c_void_p(alias_default_23.data_ptr()), c_void_p(buf767.data_ptr()))
    del alias_default_23
    del getitem_221
    buf769 = reinterpret_tensor(buf733, (6, 64, 512), (32768, 512, 1), 0); del buf733  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_69, reinterpret_tensor(buf768, (6, 512, 512), (262144, 512, 1), 0), out=buf769)
    del permute_default_69
    buf770 = reinterpret_tensor(buf729, (6, 512, 64), (32768, 64, 1), 0); del buf729  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf768, (6, 512, 512), (262144, 512, 1), 0), permute_default_70, out=buf770)
    del buf768
    del permute_default_70
    buf771 = reinterpret_tensor(buf699, (512, 384), (384, 1), 0); del buf699  # reuse
    cpp_fused_clone_151(c_void_p(buf762.data_ptr()), c_void_p(buf771.data_ptr()))
    buf772 = buf717; del buf717  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_795, reinterpret_tensor(buf771, (3072, 64, 1), (64, 1, 0), 0), out=buf772)
    del permute_795
    buf773 = buf707; del buf707  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf771, (3072, 64, 1), (64, 1, 0), 0), permute_796, out=buf773)
    del permute_796
    buf774 = buf708; del buf708  # reuse
    cpp_fused_col2im_152(c_void_p(buf774.data_ptr()))
    aten.index_put_(buf774, [None, None, unsqueeze_8, full_default_1], reinterpret_tensor(buf773, (1, 384, 9, 512, 1, 1), (0, 9, 1, 3456, 0, 0), 0), True)
    del buf773
    del full_default_1
    del unsqueeze_8
    buf777 = reinterpret_tensor(buf771, (512, 384), (1, 512), 0); del buf771  # reuse
    buf780 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_153(c_void_p(buf774.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf780.data_ptr()))
    del buf774
    buf778 = buf762; del buf762  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf777, permute_800, out=buf778)
    del permute_800
    buf779 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf777, (384, 512), (512, 1), 0), view, out=buf779)
    buf781 = reinterpret_tensor(buf767, (3072, 1, 1), (1, 3072, 3072), 0); del buf767  # reuse
    buf782 = empty((1, 1, 54), device='cpu', dtype=torch.float32)
    buf783 = buf772; del buf772  # reuse
    cpp_fused__softmax_backward_data_sum_154(c_void_p(buf783.data_ptr()), c_void_p(alias_49.data_ptr()), c_void_p(buf781.data_ptr()), c_void_p(buf782.data_ptr()))
    del alias_49
    del buf781
    buf784 = empty((54, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf783, (54, 512), (1, 54), 0), view_9, out=buf784)
    del view_9
    buf785 = reinterpret_tensor(buf777, (384, 512), (512, 1), 0); del buf777  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_9, reinterpret_tensor(buf783, (54, 512), (1, 54), 0), out=buf785)
    del buf783
    del permute_9
    buf786 = buf720; del buf720  # reuse
    buf804 = reinterpret_tensor(buf719, (512, 384), (1, 512), 0); del buf719  # reuse
    buf787 = reinterpret_tensor(buf704, (1, 384, 512), (196608, 512, 1), 0); del buf704  # reuse
    buf788 = empty((1, 384, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_transpose_view_155(c_void_p(buf785.data_ptr()), c_void_p(addmm.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(buf788.data_ptr()))
    del addmm
    del buf770
    del buf785
    del buf786
    del convolution_1
    del primals_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf789 = aten.convolution_backward(buf787, convolution, primals_25, [0], [1], [0], [1], False, [0], 1, [True, True, False])
    del convolution
    del primals_25
    buf790 = buf789[0]
    buf791 = buf789[1]
    del buf789
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf792 = aten.convolution_backward(buf790, permute_3, primals_24, [0], [1], [4], [1], False, [0], 768, [True, True, False])
    del permute_3
    del primals_24
    buf793 = buf792[0]
    buf794 = buf792[1]
    del buf792
    buf795 = reinterpret_tensor(buf787, (512, 384), (384, 1), 0); del buf787  # reuse
    cpp_fused_view_156(c_void_p(buf765.data_ptr()), c_void_p(buf795.data_ptr()))
    del buf765
    buf796 = reinterpret_tensor(buf790, (512, 768), (768, 1), 0); del buf790  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf795, permute_814, out=buf796)
    del permute_814
    buf797 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf795, (384, 512), (1, 384), 0), view, out=buf797)
    buf798 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf799 = reinterpret_tensor(buf769, (512, 384), (1, 512), 0); del buf769  # reuse
    cpp_fused_sum_view_157(c_void_p(buf799.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf798.data_ptr()))
    del buf795
    buf800 = reinterpret_tensor(buf761, (512, 768), (768, 1), 0); del buf761  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf799, permute_818, out=buf800)
    del permute_818
    buf801 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf799, (384, 512), (512, 1), 0), view, out=buf801)
    buf802 = empty((1, 384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_158(c_void_p(buf799.data_ptr()), c_void_p(buf802.data_ptr()))
    del buf799
    buf805 = reinterpret_tensor(buf745, (512, 768), (768, 1), 0); del buf745  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf804, permute_822, out=buf805)
    del permute_822
    buf806 = empty((384, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf804, (384, 512), (512, 1), 0), view, out=buf806)
    del view
    buf807 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf808 = buf758; del buf758  # reuse
    buf809 = buf757; del buf757  # reuse
    buf810 = buf756; del buf756  # reuse
    buf815 = reinterpret_tensor(buf734, (1, 512, 768), (393216, 768, 1), 0); del buf734  # reuse
    buf819 = reinterpret_tensor(buf730, (1, 512, 768), (393216, 768, 1), 0); del buf730  # reuse
    buf823 = reinterpret_tensor(buf727, (1, 512, 768), (393216, 768, 1), 0); del buf727  # reuse
    buf812 = empty((768, ), device='cpu', dtype=torch.float32)
    buf813 = empty((768, ), device='cpu', dtype=torch.float32)
    buf814 = empty((2, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_159(c_void_p(buf808.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_75.data_ptr()), c_void_p(expand.data_ptr()), c_void_p(slice_4.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()))
    del buf778
    del buf793
    del buf796
    del buf800
    del buf804
    del buf805
    del buf808
    del buf809
    del buf810
    del div_75
    del getitem_3
    del mul_1
    del primals_16
    aten.index_put_(buf814, [expand], buf815, True)
    del expand
    buf818 = reinterpret_tensor(buf815, (512, 768), (768, 1), 0); del buf815  # reuse
    cpp_fused_embedding_dense_backward_160(c_void_p(buf818.data_ptr()))
    aten.index_put_(buf818, [slice_4], buf819, True)
    del buf819
    del slice_4
    buf822 = empty((30522, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_161(c_void_p(buf822.data_ptr()))
    aten.index_put_(buf822, [primals_290], buf823, True)
    del buf823
    del primals_290
    return (reinterpret_tensor(buf788, (384, 1), (1, 1), 0), reinterpret_tensor(buf722, (384, 1), (1, 1), 0), reinterpret_tensor(buf656, (384, 1), (1, 1), 0), reinterpret_tensor(buf590, (384, 1), (1, 1), 0), reinterpret_tensor(buf524, (384, 1), (1, 1), 0), reinterpret_tensor(buf458, (384, 1), (1, 1), 0), reinterpret_tensor(buf392, (384, 1), (1, 1), 0), reinterpret_tensor(buf326, (384, 1), (1, 1), 0), reinterpret_tensor(buf260, (384, 1), (1, 1), 0), reinterpret_tensor(buf194, (384, 1), (1, 1), 0), reinterpret_tensor(buf128, (384, 1), (1, 1), 0), reinterpret_tensor(buf62, (384, 1), (1, 1), 0), buf822, buf818, buf814, buf812, buf813, reinterpret_tensor(buf806, (384, 768), (768, 1), 0), reinterpret_tensor(buf807, (384, ), (1, ), 0), reinterpret_tensor(buf801, (384, 768), (768, 1), 0), reinterpret_tensor(buf802, (384, ), (1, ), 0), reinterpret_tensor(buf797, (384, 768), (768, 1), 0), reinterpret_tensor(buf798, (384, ), (1, ), 0), buf794, buf791, reinterpret_tensor(buf784, (54, 384), (384, 1), 0), reinterpret_tensor(buf782, (54, ), (1, ), 0), reinterpret_tensor(buf779, (384, 768), (768, 1), 0), reinterpret_tensor(buf780, (384, ), (1, ), 0), reinterpret_tensor(buf763, (768, 768), (768, 1), 0), reinterpret_tensor(buf764, (768, ), (1, ), 0), buf759, buf760, reinterpret_tensor(buf754, (3072, 768), (768, 1), 0), reinterpret_tensor(buf755, (3072, ), (1, ), 0), reinterpret_tensor(buf750, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf751, (768, ), (1, ), 0), buf746, buf747, reinterpret_tensor(buf740, (384, 768), (768, 1), 0), reinterpret_tensor(buf741, (384, ), (1, ), 0), reinterpret_tensor(buf735, (384, 768), (768, 1), 0), reinterpret_tensor(buf736, (384, ), (1, ), 0), reinterpret_tensor(buf731, (384, 768), (768, 1), 0), reinterpret_tensor(buf732, (384, ), (1, ), 0), buf728, buf725, reinterpret_tensor(buf718, (54, 384), (384, 1), 0), reinterpret_tensor(buf716, (54, ), (1, ), 0), reinterpret_tensor(buf713, (384, 768), (768, 1), 0), reinterpret_tensor(buf714, (384, ), (1, ), 0), reinterpret_tensor(buf697, (768, 768), (768, 1), 0), reinterpret_tensor(buf698, (768, ), (1, ), 0), buf693, buf694, reinterpret_tensor(buf688, (3072, 768), (768, 1), 0), reinterpret_tensor(buf689, (3072, ), (1, ), 0), reinterpret_tensor(buf684, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf685, (768, ), (1, ), 0), buf680, buf681, reinterpret_tensor(buf674, (384, 768), (768, 1), 0), reinterpret_tensor(buf675, (384, ), (1, ), 0), reinterpret_tensor(buf669, (384, 768), (768, 1), 0), reinterpret_tensor(buf670, (384, ), (1, ), 0), reinterpret_tensor(buf665, (384, 768), (768, 1), 0), reinterpret_tensor(buf666, (384, ), (1, ), 0), buf662, buf659, reinterpret_tensor(buf652, (54, 384), (384, 1), 0), reinterpret_tensor(buf650, (54, ), (1, ), 0), reinterpret_tensor(buf647, (384, 768), (768, 1), 0), reinterpret_tensor(buf648, (384, ), (1, ), 0), reinterpret_tensor(buf631, (768, 768), (768, 1), 0), reinterpret_tensor(buf632, (768, ), (1, ), 0), buf627, buf628, reinterpret_tensor(buf622, (3072, 768), (768, 1), 0), reinterpret_tensor(buf623, (3072, ), (1, ), 0), reinterpret_tensor(buf618, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf619, (768, ), (1, ), 0), buf614, buf615, reinterpret_tensor(buf608, (384, 768), (768, 1), 0), reinterpret_tensor(buf609, (384, ), (1, ), 0), reinterpret_tensor(buf603, (384, 768), (768, 1), 0), reinterpret_tensor(buf604, (384, ), (1, ), 0), reinterpret_tensor(buf599, (384, 768), (768, 1), 0), reinterpret_tensor(buf600, (384, ), (1, ), 0), buf596, buf593, reinterpret_tensor(buf586, (54, 384), (384, 1), 0), reinterpret_tensor(buf584, (54, ), (1, ), 0), reinterpret_tensor(buf581, (384, 768), (768, 1), 0), reinterpret_tensor(buf582, (384, ), (1, ), 0), reinterpret_tensor(buf565, (768, 768), (768, 1), 0), reinterpret_tensor(buf566, (768, ), (1, ), 0), buf561, buf562, reinterpret_tensor(buf556, (3072, 768), (768, 1), 0), reinterpret_tensor(buf557, (3072, ), (1, ), 0), reinterpret_tensor(buf552, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf553, (768, ), (1, ), 0), buf548, buf549, reinterpret_tensor(buf542, (384, 768), (768, 1), 0), reinterpret_tensor(buf543, (384, ), (1, ), 0), reinterpret_tensor(buf537, (384, 768), (768, 1), 0), reinterpret_tensor(buf538, (384, ), (1, ), 0), reinterpret_tensor(buf533, (384, 768), (768, 1), 0), reinterpret_tensor(buf534, (384, ), (1, ), 0), buf530, buf527, reinterpret_tensor(buf520, (54, 384), (384, 1), 0), reinterpret_tensor(buf518, (54, ), (1, ), 0), reinterpret_tensor(buf515, (384, 768), (768, 1), 0), reinterpret_tensor(buf516, (384, ), (1, ), 0), reinterpret_tensor(buf499, (768, 768), (768, 1), 0), reinterpret_tensor(buf500, (768, ), (1, ), 0), buf495, buf496, reinterpret_tensor(buf490, (3072, 768), (768, 1), 0), reinterpret_tensor(buf491, (3072, ), (1, ), 0), reinterpret_tensor(buf486, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf487, (768, ), (1, ), 0), buf482, buf483, reinterpret_tensor(buf476, (384, 768), (768, 1), 0), reinterpret_tensor(buf477, (384, ), (1, ), 0), reinterpret_tensor(buf471, (384, 768), (768, 1), 0), reinterpret_tensor(buf472, (384, ), (1, ), 0), reinterpret_tensor(buf467, (384, 768), (768, 1), 0), reinterpret_tensor(buf468, (384, ), (1, ), 0), buf464, buf461, reinterpret_tensor(buf454, (54, 384), (384, 1), 0), reinterpret_tensor(buf452, (54, ), (1, ), 0), reinterpret_tensor(buf449, (384, 768), (768, 1), 0), reinterpret_tensor(buf450, (384, ), (1, ), 0), reinterpret_tensor(buf433, (768, 768), (768, 1), 0), reinterpret_tensor(buf434, (768, ), (1, ), 0), buf429, buf430, reinterpret_tensor(buf424, (3072, 768), (768, 1), 0), reinterpret_tensor(buf425, (3072, ), (1, ), 0), reinterpret_tensor(buf420, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf421, (768, ), (1, ), 0), buf416, buf417, reinterpret_tensor(buf410, (384, 768), (768, 1), 0), reinterpret_tensor(buf411, (384, ), (1, ), 0), reinterpret_tensor(buf405, (384, 768), (768, 1), 0), reinterpret_tensor(buf406, (384, ), (1, ), 0), reinterpret_tensor(buf401, (384, 768), (768, 1), 0), reinterpret_tensor(buf402, (384, ), (1, ), 0), buf398, buf395, reinterpret_tensor(buf388, (54, 384), (384, 1), 0), reinterpret_tensor(buf386, (54, ), (1, ), 0), reinterpret_tensor(buf383, (384, 768), (768, 1), 0), reinterpret_tensor(buf384, (384, ), (1, ), 0), reinterpret_tensor(buf367, (768, 768), (768, 1), 0), reinterpret_tensor(buf368, (768, ), (1, ), 0), buf363, buf364, reinterpret_tensor(buf358, (3072, 768), (768, 1), 0), reinterpret_tensor(buf359, (3072, ), (1, ), 0), reinterpret_tensor(buf354, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf355, (768, ), (1, ), 0), buf350, buf351, reinterpret_tensor(buf344, (384, 768), (768, 1), 0), reinterpret_tensor(buf345, (384, ), (1, ), 0), reinterpret_tensor(buf339, (384, 768), (768, 1), 0), reinterpret_tensor(buf340, (384, ), (1, ), 0), reinterpret_tensor(buf335, (384, 768), (768, 1), 0), reinterpret_tensor(buf336, (384, ), (1, ), 0), buf332, buf329, reinterpret_tensor(buf322, (54, 384), (384, 1), 0), reinterpret_tensor(buf320, (54, ), (1, ), 0), reinterpret_tensor(buf317, (384, 768), (768, 1), 0), reinterpret_tensor(buf318, (384, ), (1, ), 0), reinterpret_tensor(buf301, (768, 768), (768, 1), 0), reinterpret_tensor(buf302, (768, ), (1, ), 0), buf297, buf298, reinterpret_tensor(buf292, (3072, 768), (768, 1), 0), reinterpret_tensor(buf293, (3072, ), (1, ), 0), reinterpret_tensor(buf288, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf289, (768, ), (1, ), 0), buf284, buf285, reinterpret_tensor(buf278, (384, 768), (768, 1), 0), reinterpret_tensor(buf279, (384, ), (1, ), 0), reinterpret_tensor(buf273, (384, 768), (768, 1), 0), reinterpret_tensor(buf274, (384, ), (1, ), 0), reinterpret_tensor(buf269, (384, 768), (768, 1), 0), reinterpret_tensor(buf270, (384, ), (1, ), 0), buf266, buf263, reinterpret_tensor(buf256, (54, 384), (384, 1), 0), reinterpret_tensor(buf254, (54, ), (1, ), 0), reinterpret_tensor(buf251, (384, 768), (768, 1), 0), reinterpret_tensor(buf252, (384, ), (1, ), 0), reinterpret_tensor(buf235, (768, 768), (768, 1), 0), reinterpret_tensor(buf236, (768, ), (1, ), 0), buf231, buf232, reinterpret_tensor(buf226, (3072, 768), (768, 1), 0), reinterpret_tensor(buf227, (3072, ), (1, ), 0), reinterpret_tensor(buf222, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf223, (768, ), (1, ), 0), buf218, buf219, reinterpret_tensor(buf212, (384, 768), (768, 1), 0), reinterpret_tensor(buf213, (384, ), (1, ), 0), reinterpret_tensor(buf207, (384, 768), (768, 1), 0), reinterpret_tensor(buf208, (384, ), (1, ), 0), reinterpret_tensor(buf203, (384, 768), (768, 1), 0), reinterpret_tensor(buf204, (384, ), (1, ), 0), buf200, buf197, reinterpret_tensor(buf190, (54, 384), (384, 1), 0), reinterpret_tensor(buf188, (54, ), (1, ), 0), reinterpret_tensor(buf185, (384, 768), (768, 1), 0), reinterpret_tensor(buf186, (384, ), (1, ), 0), reinterpret_tensor(buf169, (768, 768), (768, 1), 0), reinterpret_tensor(buf170, (768, ), (1, ), 0), buf165, buf166, reinterpret_tensor(buf160, (3072, 768), (768, 1), 0), reinterpret_tensor(buf161, (3072, ), (1, ), 0), reinterpret_tensor(buf156, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf157, (768, ), (1, ), 0), buf152, buf153, reinterpret_tensor(buf146, (384, 768), (768, 1), 0), reinterpret_tensor(buf147, (384, ), (1, ), 0), reinterpret_tensor(buf141, (384, 768), (768, 1), 0), reinterpret_tensor(buf142, (384, ), (1, ), 0), reinterpret_tensor(buf137, (384, 768), (768, 1), 0), reinterpret_tensor(buf138, (384, ), (1, ), 0), buf134, buf131, reinterpret_tensor(buf124, (54, 384), (384, 1), 0), reinterpret_tensor(buf122, (54, ), (1, ), 0), reinterpret_tensor(buf119, (384, 768), (768, 1), 0), reinterpret_tensor(buf120, (384, ), (1, ), 0), reinterpret_tensor(buf103, (768, 768), (768, 1), 0), reinterpret_tensor(buf104, (768, ), (1, ), 0), buf99, buf100, reinterpret_tensor(buf94, (3072, 768), (768, 1), 0), reinterpret_tensor(buf95, (3072, ), (1, ), 0), reinterpret_tensor(buf90, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf91, (768, ), (1, ), 0), buf86, buf87, reinterpret_tensor(buf80, (384, 768), (768, 1), 0), reinterpret_tensor(buf81, (384, ), (1, ), 0), reinterpret_tensor(buf75, (384, 768), (768, 1), 0), reinterpret_tensor(buf76, (384, ), (1, ), 0), reinterpret_tensor(buf71, (384, 768), (768, 1), 0), reinterpret_tensor(buf72, (384, ), (1, ), 0), buf68, buf65, reinterpret_tensor(buf58, (54, 384), (384, 1), 0), reinterpret_tensor(buf56, (54, ), (1, ), 0), reinterpret_tensor(buf53, (384, 768), (768, 1), 0), reinterpret_tensor(buf54, (384, ), (1, ), 0), reinterpret_tensor(buf37, (768, 768), (768, 1), 0), reinterpret_tensor(buf38, (768, ), (1, ), 0), buf33, buf34, reinterpret_tensor(buf28, (3072, 768), (768, 1), 0), reinterpret_tensor(buf29, (3072, ), (1, ), 0), reinterpret_tensor(buf24, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf25, (768, ), (1, ), 0), buf20, buf21, reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf16, (768, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf7, (30522, 768), (768, 1), 0), reinterpret_tensor(buf8, (30522, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_291 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_4 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_3 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_9 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    full_default_1 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    unsqueeze_8 = rand_strided((9, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.int64)
    getitem_221 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_67 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_68 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_23 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_69 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_70 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_30 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_9 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_36 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_22 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_28 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_219 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_61 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_62 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_21 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_63 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_64 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_12 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_12 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_70 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_72 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_41 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_47 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_217 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_55 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_56 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_19 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_57 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_58 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_102 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_20 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_21 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_60 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_66 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_215 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_49 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_50 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_17 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_51 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_52 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_138 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_28 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_142 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_33 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_144 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_79 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_85 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_153 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_213 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_43 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_44 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_15 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_45 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_46 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_33 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_180 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_35 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_98 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_189 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_211 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_37 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_38 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_13 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_39 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_40 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_210 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_44 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_212 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_225 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_209 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_31 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_32 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_11 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_33 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_34 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_246 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_248 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_47 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_250 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_252 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_49 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_261 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_207 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_25 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_26 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_9 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_27 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_28 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_282 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_60 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_284 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_54 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_286 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_65 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_288 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_56 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_297 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_205 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_19 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_20 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_7 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_21 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_22 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_318 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_68 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_320 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_61 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_322 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_324 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_63 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_174 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_180 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_333 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_203 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_13 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_14 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_5 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_15 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_16 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_354 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_76 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_356 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_68 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_358 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_360 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_193 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_369 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_201 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_7 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_8 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_3 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_9 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_10 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_390 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_84 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_392 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_75 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_394 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_89 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_396 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_77 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((1, 768, 512), (393216, 512, 1), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((1, 384, 512), (196608, 512, 1), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((384, 54), (1, 384), device='cpu', dtype=torch.float32)
    view_405 = rand_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    getitem_199 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    permute_default_1 = rand_strided((6, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_2 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_1 = rand_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    permute_default_3 = rand_strided((6, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_4 = rand_strided((6, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_426 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_92 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_428 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_82 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_430 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_432 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_84 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_102 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_434 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_52 = rand_strided((512, 30522), (30522, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_230 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_38 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_242 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_246 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_256 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_257 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_27 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_275 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_279 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_287 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_295 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_305 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_306 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_29 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_324 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_328 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_332 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_336 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_354 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_359 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_31 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_373 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_385 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_389 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_403 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_404 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_33 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_422 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_426 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_430 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_442 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_452 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_453 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_457 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_35 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_479 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_483 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_487 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_491 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_501 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_502 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_506 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_37 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_520 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_524 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_528 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_532 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_536 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_540 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_550 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_551 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_555 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_39 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_569 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_573 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_577 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_581 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_585 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_589 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_599 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_600 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_604 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_41 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_618 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_622 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_626 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_630 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_634 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_638 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_648 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_649 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_653 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_667 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_671 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_675 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_679 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_683 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_67 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_687 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_697 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_698 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_702 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_45 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_716 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_720 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_724 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_69 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_728 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_732 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_70 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_736 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_746 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_747 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_751 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_47 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_765 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_769 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_773 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_72 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_777 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_781 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_73 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_785 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_795 = rand_strided((3072, 9, 64), (576, 1, 9), device='cpu', dtype=torch.float32)
    permute_796 = rand_strided((3072, 1, 9), (9, 27648, 1), device='cpu', dtype=torch.float32)
    permute_800 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_49 = rand_strided((3072, 9, 1), (9, 1, 27648), device='cpu', dtype=torch.float32)
    permute_814 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_818 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_822 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_75 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 30522), (15627264, 30522, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_16, primals_24, primals_25, primals_32, primals_38, primals_46, primals_47, primals_54, primals_60, primals_68, primals_69, primals_76, primals_82, primals_90, primals_91, primals_98, primals_104, primals_112, primals_113, primals_120, primals_126, primals_134, primals_135, primals_142, primals_148, primals_156, primals_157, primals_164, primals_170, primals_178, primals_179, primals_186, primals_192, primals_200, primals_201, primals_208, primals_214, primals_222, primals_223, primals_230, primals_236, primals_244, primals_245, primals_252, primals_258, primals_266, primals_267, primals_274, primals_280, primals_284, primals_290, primals_291, expand, slice_4, mul_1, getitem_3, view, addmm, permute_3, convolution, convolution_1, permute_9, view_9, full_default_1, unsqueeze_8, getitem_221, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_30, getitem_7, mul_4, view_32, addmm_5, view_34, getitem_11, mul_9, view_36, addmm_7, permute_22, convolution_2, convolution_3, permute_28, view_45, getitem_219, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_66, getitem_17, mul_12, view_68, addmm_12, view_70, getitem_21, mul_17, view_72, addmm_14, permute_41, convolution_4, convolution_5, permute_47, view_81, getitem_217, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_102, getitem_27, mul_20, view_104, addmm_19, view_106, getitem_31, mul_25, view_108, addmm_21, permute_60, convolution_6, convolution_7, permute_66, view_117, getitem_215, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_138, getitem_37, mul_28, view_140, addmm_26, view_142, getitem_41, mul_33, view_144, addmm_28, permute_79, convolution_8, convolution_9, permute_85, view_153, getitem_213, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_174, getitem_47, mul_36, view_176, addmm_33, view_178, getitem_51, mul_41, view_180, addmm_35, permute_98, convolution_10, convolution_11, permute_104, view_189, getitem_211, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_210, getitem_57, mul_44, view_212, addmm_40, view_214, getitem_61, mul_49, view_216, addmm_42, permute_117, convolution_12, convolution_13, permute_123, view_225, getitem_209, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_246, getitem_67, mul_52, view_248, addmm_47, view_250, getitem_71, mul_57, view_252, addmm_49, permute_136, convolution_14, convolution_15, permute_142, view_261, getitem_207, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_282, getitem_77, mul_60, view_284, addmm_54, view_286, getitem_81, mul_65, view_288, addmm_56, permute_155, convolution_16, convolution_17, permute_161, view_297, getitem_205, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_318, getitem_87, mul_68, view_320, addmm_61, view_322, getitem_91, mul_73, view_324, addmm_63, permute_174, convolution_18, convolution_19, permute_180, view_333, getitem_203, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_354, getitem_97, mul_76, view_356, addmm_68, view_358, getitem_101, mul_81, view_360, addmm_70, permute_193, convolution_20, convolution_21, permute_199, view_369, getitem_201, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_390, getitem_107, mul_84, view_392, addmm_75, view_394, getitem_111, mul_89, view_396, addmm_77, permute_212, convolution_22, convolution_23, permute_218, view_405, getitem_199, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_426, getitem_117, mul_92, view_428, addmm_82, view_430, getitem_121, mul_97, view_432, addmm_84, mul_102, view_434, sub_52, convert_element_type, permute_230, div_38, permute_234, div_39, permute_238, permute_242, div_40, permute_246, permute_256, permute_257, permute_261, alias_27, permute_275, permute_279, permute_283, div_42, permute_287, permute_291, div_43, permute_295, permute_305, permute_306, permute_310, alias_29, permute_324, permute_328, permute_332, div_45, permute_336, permute_340, div_46, permute_344, permute_354, permute_355, permute_359, alias_31, permute_373, permute_377, permute_381, div_48, permute_385, permute_389, div_49, permute_393, permute_403, permute_404, permute_408, alias_33, permute_422, permute_426, permute_430, div_51, permute_434, permute_438, div_52, permute_442, permute_452, permute_453, permute_457, alias_35, permute_471, permute_475, permute_479, div_54, permute_483, permute_487, div_55, permute_491, permute_501, permute_502, permute_506, alias_37, permute_520, permute_524, permute_528, div_57, permute_532, permute_536, div_58, permute_540, permute_550, permute_551, permute_555, alias_39, permute_569, permute_573, permute_577, div_60, permute_581, permute_585, div_61, permute_589, permute_599, permute_600, permute_604, alias_41, permute_618, permute_622, permute_626, div_63, permute_630, permute_634, div_64, permute_638, permute_648, permute_649, permute_653, alias_43, permute_667, permute_671, permute_675, div_66, permute_679, permute_683, div_67, permute_687, permute_697, permute_698, permute_702, alias_45, permute_716, permute_720, permute_724, div_69, permute_728, permute_732, div_70, permute_736, permute_746, permute_747, permute_751, alias_47, permute_765, permute_769, permute_773, div_72, permute_777, permute_781, div_73, permute_785, permute_795, permute_796, permute_800, alias_49, permute_814, permute_818, permute_822, div_75, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('YituTechConvBert', benchmark_compiled_module)
