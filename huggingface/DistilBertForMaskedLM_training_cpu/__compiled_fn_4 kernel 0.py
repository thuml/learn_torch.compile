
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3906816L); x0+=static_cast<long>(8L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (30522L*x1))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_gelu_backward_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
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
        }
    }
}
''')


cpp_fused_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr6[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
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
        }
    }
}
''')


cpp_fused_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr6[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
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
        }
    }
}
''')


cpp_fused_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr6[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
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
        }
    }
}
''')


cpp_fused_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr6[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
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
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr6[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
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
        }
    }
}
''')


cpp_fused_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_56 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp25 = static_cast<int>(0);
                auto tmp26 = tmp24 == tmp25;
                auto tmp27 = to_float_mask(tmp26);
                auto tmp28 = decltype(tmp22)::blendv(tmp16, tmp22, tmp27);
                tmp23.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                tmp28.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr7 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
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


cpp_fused_embedding_dense_backward_57 = async_compile.cpp('''
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
    primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_103, primals_108, primals_109, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, view_138, addmm_36, mul_47, view_140, sub_21, convert_element_type, permute_68, div_14, permute_72, div_15, permute_76, permute_80, div_16, permute_84, permute_89, permute_90, alias_8, permute_91, permute_92, permute_95, permute_100, permute_105, div_18, permute_109, permute_113, div_19, permute_117, permute_122, permute_123, alias_9, permute_124, permute_125, permute_128, permute_133, permute_138, div_21, permute_142, permute_146, div_22, permute_150, permute_155, permute_156, alias_10, permute_157, permute_158, permute_161, permute_166, permute_171, div_24, permute_175, permute_179, div_25, permute_183, permute_188, permute_189, alias_11, permute_190, permute_191, permute_194, permute_199, permute_204, div_27, permute_208, permute_212, div_28, permute_216, permute_221, permute_222, alias_12, permute_223, permute_224, permute_227, permute_232, permute_237, div_30, permute_241, permute_245, div_31, permute_249, permute_254, permute_255, alias_13, permute_256, permute_257, permute_260, permute_265, permute_270, div_33, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_108, (1, 128), (128, 1))
    assert_size_stride(primals_109, (1, 128), (128, 1))
    assert_size_stride(slice_2, (1, 128), (512, 1))
    assert_size_stride(mul, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(getitem_3, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view, (128, 768), (768, 1))
    assert_size_stride(view_12, (1, 1, 1, 128), (128, 128, 128, 1))
    assert_size_stride(getitem_5, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_17, (128, 768), (768, 1))
    assert_size_stride(mul_2, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_19, (128, 768), (768, 1))
    assert_size_stride(addmm_4, (128, 3072), (3072, 1))
    assert_size_stride(view_21, (128, 3072), (3072, 1))
    assert_size_stride(getitem_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_7, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_23, (128, 768), (768, 1))
    assert_size_stride(getitem_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_40, (128, 768), (768, 1))
    assert_size_stride(mul_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_42, (128, 768), (768, 1))
    assert_size_stride(addmm_10, (128, 3072), (3072, 1))
    assert_size_stride(view_44, (128, 3072), (3072, 1))
    assert_size_stride(getitem_17, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_14, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_46, (128, 768), (768, 1))
    assert_size_stride(getitem_21, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_63, (128, 768), (768, 1))
    assert_size_stride(mul_16, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_65, (128, 768), (768, 1))
    assert_size_stride(addmm_16, (128, 3072), (3072, 1))
    assert_size_stride(view_67, (128, 3072), (3072, 1))
    assert_size_stride(getitem_25, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_21, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_69, (128, 768), (768, 1))
    assert_size_stride(getitem_29, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_86, (128, 768), (768, 1))
    assert_size_stride(mul_23, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_88, (128, 768), (768, 1))
    assert_size_stride(addmm_22, (128, 3072), (3072, 1))
    assert_size_stride(view_90, (128, 3072), (3072, 1))
    assert_size_stride(getitem_33, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_28, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_92, (128, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_109, (128, 768), (768, 1))
    assert_size_stride(mul_30, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_111, (128, 768), (768, 1))
    assert_size_stride(addmm_28, (128, 3072), (3072, 1))
    assert_size_stride(view_113, (128, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_35, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_115, (128, 768), (768, 1))
    assert_size_stride(getitem_45, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_132, (128, 768), (768, 1))
    assert_size_stride(mul_37, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_134, (128, 768), (768, 1))
    assert_size_stride(addmm_34, (128, 3072), (3072, 1))
    assert_size_stride(view_136, (128, 3072), (3072, 1))
    assert_size_stride(getitem_49, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_42, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_138, (128, 768), (768, 1))
    assert_size_stride(addmm_36, (128, 768), (768, 1))
    assert_size_stride(mul_47, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_140, (128, 768), (768, 1))
    assert_size_stride(sub_21, (128, 30522), (30522, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_68, (30522, 768), (768, 1))
    assert_size_stride(div_14, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_72, (768, 768), (768, 1))
    assert_size_stride(div_15, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_76, (768, 3072), (3072, 1))
    assert_size_stride(permute_80, (3072, 768), (768, 1))
    assert_size_stride(div_16, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_84, (768, 768), (768, 1))
    assert_size_stride(permute_89, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_90, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_8, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_91, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_92, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_95, (768, 768), (768, 1))
    assert_size_stride(permute_100, (768, 768), (768, 1))
    assert_size_stride(permute_105, (768, 768), (768, 1))
    assert_size_stride(div_18, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_109, (768, 3072), (3072, 1))
    assert_size_stride(permute_113, (3072, 768), (768, 1))
    assert_size_stride(div_19, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_117, (768, 768), (768, 1))
    assert_size_stride(permute_122, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_123, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_9, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_124, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_125, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_128, (768, 768), (768, 1))
    assert_size_stride(permute_133, (768, 768), (768, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(div_21, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_142, (768, 3072), (3072, 1))
    assert_size_stride(permute_146, (3072, 768), (768, 1))
    assert_size_stride(div_22, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_150, (768, 768), (768, 1))
    assert_size_stride(permute_155, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_156, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_10, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_157, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_158, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_166, (768, 768), (768, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(div_24, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(div_25, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_188, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_189, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_11, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_190, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_191, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_199, (768, 768), (768, 1))
    assert_size_stride(permute_204, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_221, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_222, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_12, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_223, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_224, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_227, (768, 768), (768, 1))
    assert_size_stride(permute_232, (768, 768), (768, 1))
    assert_size_stride(permute_237, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_241, (768, 3072), (3072, 1))
    assert_size_stride(permute_245, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_249, (768, 768), (768, 1))
    assert_size_stride(permute_254, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_255, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_256, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_257, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_260, (768, 768), (768, 1))
    assert_size_stride(permute_265, (768, 768), (768, 1))
    assert_size_stride(permute_270, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 30522), (3906816, 30522, 1))
    buf0 = empty((128, 30522), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_109.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((128, 30522), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 128, 30522), (3906816, 30522, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_21.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del convert_element_type
    del primals_109
    del sub_21
    del tangents_1
    del tangents_2
    buf6 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (128, 30522), (30522, 1), 0), permute_68, out=buf6)
    del permute_68
    buf7 = empty((30522, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (30522, 128), (1, 30522), 0), view_140, out=buf7)
    del view_140
    buf8 = empty((1, 30522), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf4, (1, 128, 1), (128, 1, 128), 0); del buf4  # reuse
    buf10 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf11 = empty((768, ), device='cpu', dtype=torch.float32)
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf6, (1, 128, 768), (98304, 768, 1), 0); del buf6  # reuse
    cpp_fused_gelu_gelu_backward_native_layer_norm_backward_sum_2(c_void_p(buf13.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(mul_47.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(addmm_36.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del addmm_36
    del buf5
    del div_14
    del mul_47
    del primals_103
    buf14 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (128, 768), (768, 1), 0), permute_72, out=buf14)
    del permute_72
    buf15 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (768, 128), (1, 768), 0), view_138, out=buf15)
    del view_138
    buf16 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf17 = buf9; del buf9  # reuse
    buf18 = buf10; del buf10  # reuse
    buf19 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf20 = empty((768, ), device='cpu', dtype=torch.float32)
    buf21 = empty((768, ), device='cpu', dtype=torch.float32)
    buf22 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del div_15
    del getitem_49
    del mul_42
    del primals_99
    buf23 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (128, 768), (768, 1), 0), permute_76, out=buf23)
    del permute_76
    buf24 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (768, 128), (1, 768), 0), view_136, out=buf24)
    del view_136
    buf25 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf23, (1, 128, 3072), (393216, 3072, 1), 0); del buf23  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf26.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf25.data_ptr()))
    del addmm_34
    buf27 = reinterpret_tensor(buf22, (128, 768), (768, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (128, 3072), (3072, 1), 0), permute_80, out=buf27)
    del permute_80
    buf28 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (3072, 128), (1, 3072), 0), view_134, out=buf28)
    del view_134
    buf29 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf30 = buf18; del buf18  # reuse
    buf31 = buf17; del buf17  # reuse
    buf32 = reinterpret_tensor(buf14, (1, 128, 768), (98304, 768, 1), 0); del buf14  # reuse
    buf33 = empty((768, ), device='cpu', dtype=torch.float32)
    buf34 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_5(c_void_p(buf26.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(mul_37.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del div_16
    del mul_37
    del primals_93
    buf35 = buf27; del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (128, 768), (768, 1), 0), permute_84, out=buf35)
    del permute_84
    buf36 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (768, 128), (1, 768), 0), view_132, out=buf36)
    del view_132
    buf38 = reinterpret_tensor(buf19, (12, 128, 64), (8192, 64, 1), 0); del buf19  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_89, reinterpret_tensor(buf35, (12, 128, 64), (64, 768, 1), 0), out=buf38)
    del permute_89
    buf44 = reinterpret_tensor(buf13, (128, 768), (768, 1), 0); del buf13  # reuse
    cpp_fused_view_6(c_void_p(buf38.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = reinterpret_tensor(buf38, (128, 768), (768, 1), 0); del buf38  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf44, permute_95, out=buf45)
    del permute_95
    buf39 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf35, (12, 128, 64), (64, 768, 1), 0), permute_90, out=buf39)
    del permute_90
    buf40 = empty_strided((1, 12, 128, 1), (1536, 128, 1, 1536), device='cpu', dtype=torch.float32)
    buf41 = reinterpret_tensor(buf39, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf39  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_7(c_void_p(buf41.data_ptr()), c_void_p(getitem_45.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf40.data_ptr()))
    del alias_8
    del getitem_45
    buf42 = reinterpret_tensor(buf35, (12, 64, 128), (8192, 128, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_91, reinterpret_tensor(buf41, (12, 128, 128), (16384, 128, 1), 0), out=buf42)
    del permute_91
    buf48 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (128, 768), (1, 128), 0), permute_100, out=buf48)
    del permute_100
    buf43 = empty((12, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf41, (12, 128, 128), (16384, 128, 1), 0), permute_92, out=buf43)
    del permute_92
    buf51 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_8(c_void_p(buf43.data_ptr()), c_void_p(buf51.data_ptr()))
    buf52 = reinterpret_tensor(buf43, (128, 768), (768, 1), 0); del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf51, permute_105, out=buf52)
    del permute_105
    buf37 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf59 = empty((768, ), device='cpu', dtype=torch.float32)
    buf60 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_9(c_void_p(buf32.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    buf46 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (768, 128), (1, 768), 0), view_115, out=buf46)
    buf47 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_10(c_void_p(buf44.data_ptr()), c_void_p(buf47.data_ptr()))
    buf49 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (768, 128), (128, 1), 0), view_115, out=buf49)
    buf50 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_11(c_void_p(buf42.data_ptr()), c_void_p(buf50.data_ptr()))
    buf53 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (768, 128), (1, 768), 0), view_115, out=buf53)
    del view_115
    buf54 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf55 = buf32; del buf32  # reuse
    buf56 = buf31; del buf31  # reuse
    buf57 = buf30; del buf30  # reuse
    buf58 = buf55; del buf55  # reuse
    buf61 = reinterpret_tensor(buf42, (1, 128, 768), (98304, 768, 1), 0); del buf42  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_12(c_void_p(buf58.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf61.data_ptr()))
    del div_18
    del getitem_41
    del mul_35
    del primals_83
    buf62 = reinterpret_tensor(buf26, (128, 3072), (3072, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (128, 768), (768, 1), 0), permute_109, out=buf62)
    del permute_109
    buf63 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (768, 128), (1, 768), 0), view_113, out=buf63)
    del view_113
    buf64 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf65 = reinterpret_tensor(buf62, (1, 128, 3072), (393216, 3072, 1), 0); del buf62  # reuse
    cpp_fused_gelu_gelu_backward_sum_13(c_void_p(buf65.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf64.data_ptr()))
    del addmm_28
    buf66 = reinterpret_tensor(buf61, (128, 768), (768, 1), 0); del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf65, (128, 3072), (3072, 1), 0), permute_113, out=buf66)
    del permute_113
    buf67 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf65, (3072, 128), (1, 3072), 0), view_111, out=buf67)
    del view_111
    buf68 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf69 = buf57; del buf57  # reuse
    buf70 = buf56; del buf56  # reuse
    buf71 = reinterpret_tensor(buf52, (1, 128, 768), (98304, 768, 1), 0); del buf52  # reuse
    buf72 = empty((768, ), device='cpu', dtype=torch.float32)
    buf73 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_14(c_void_p(buf65.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(mul_30.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del div_19
    del mul_30
    del primals_77
    buf74 = buf66; del buf66  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (128, 768), (768, 1), 0), permute_117, out=buf74)
    del permute_117
    buf75 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (768, 128), (1, 768), 0), view_109, out=buf75)
    del view_109
    buf77 = reinterpret_tensor(buf58, (12, 128, 64), (8192, 64, 1), 0); del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_122, reinterpret_tensor(buf74, (12, 128, 64), (64, 768, 1), 0), out=buf77)
    del permute_122
    buf83 = buf51; del buf51  # reuse
    cpp_fused_view_15(c_void_p(buf77.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = reinterpret_tensor(buf77, (128, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf83, permute_128, out=buf84)
    del permute_128
    buf78 = reinterpret_tensor(buf41, (12, 128, 128), (16384, 128, 1), 0); del buf41  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf74, (12, 128, 64), (64, 768, 1), 0), permute_123, out=buf78)
    del permute_123
    buf79 = buf40; del buf40  # reuse
    buf80 = reinterpret_tensor(buf78, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf78  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_16(c_void_p(buf80.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf79.data_ptr()))
    del alias_9
    del getitem_37
    buf81 = reinterpret_tensor(buf74, (12, 64, 128), (8192, 128, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_124, reinterpret_tensor(buf80, (12, 128, 128), (16384, 128, 1), 0), out=buf81)
    del permute_124
    buf87 = buf48; del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf81, (128, 768), (1, 128), 0), permute_133, out=buf87)
    del permute_133
    buf82 = reinterpret_tensor(buf45, (12, 128, 64), (8192, 64, 1), 0); del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf80, (12, 128, 128), (16384, 128, 1), 0), permute_125, out=buf82)
    del permute_125
    buf90 = buf44; del buf44  # reuse
    cpp_fused_view_17(c_void_p(buf82.data_ptr()), c_void_p(buf90.data_ptr()))
    buf91 = reinterpret_tensor(buf82, (128, 768), (768, 1), 0); del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf90, permute_138, out=buf91)
    del permute_138
    buf76 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf98 = empty((768, ), device='cpu', dtype=torch.float32)
    buf99 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_18(c_void_p(buf71.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf85 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (768, 128), (1, 768), 0), view_92, out=buf85)
    buf86 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_19(c_void_p(buf83.data_ptr()), c_void_p(buf86.data_ptr()))
    buf88 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf81, (768, 128), (128, 1), 0), view_92, out=buf88)
    buf89 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_20(c_void_p(buf81.data_ptr()), c_void_p(buf89.data_ptr()))
    buf92 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (768, 128), (1, 768), 0), view_92, out=buf92)
    del view_92
    buf93 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf94 = buf71; del buf71  # reuse
    buf95 = buf70; del buf70  # reuse
    buf96 = buf69; del buf69  # reuse
    buf97 = buf94; del buf94  # reuse
    buf100 = reinterpret_tensor(buf81, (1, 128, 768), (98304, 768, 1), 0); del buf81  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21(c_void_p(buf97.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(getitem_33.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf100.data_ptr()))
    del div_21
    del getitem_33
    del mul_28
    del primals_67
    buf101 = reinterpret_tensor(buf65, (128, 3072), (3072, 1), 0); del buf65  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (128, 768), (768, 1), 0), permute_142, out=buf101)
    del permute_142
    buf102 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (768, 128), (1, 768), 0), view_90, out=buf102)
    del view_90
    buf103 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf104 = reinterpret_tensor(buf101, (1, 128, 3072), (393216, 3072, 1), 0); del buf101  # reuse
    cpp_fused_gelu_gelu_backward_sum_22(c_void_p(buf104.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf103.data_ptr()))
    del addmm_22
    buf105 = reinterpret_tensor(buf100, (128, 768), (768, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (128, 3072), (3072, 1), 0), permute_146, out=buf105)
    del permute_146
    buf106 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (3072, 128), (1, 3072), 0), view_88, out=buf106)
    del view_88
    buf107 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf108 = buf96; del buf96  # reuse
    buf109 = buf95; del buf95  # reuse
    buf110 = reinterpret_tensor(buf91, (1, 128, 768), (98304, 768, 1), 0); del buf91  # reuse
    buf111 = empty((768, ), device='cpu', dtype=torch.float32)
    buf112 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_23(c_void_p(buf104.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(mul_23.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del div_22
    del mul_23
    del primals_61
    buf113 = reinterpret_tensor(buf97, (128, 768), (768, 1), 0); del buf97  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (128, 768), (768, 1), 0), permute_150, out=buf113)
    del permute_150
    buf114 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (768, 128), (1, 768), 0), view_86, out=buf114)
    del view_86
    buf116 = reinterpret_tensor(buf105, (12, 128, 64), (8192, 64, 1), 0); del buf105  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_155, reinterpret_tensor(buf113, (12, 128, 64), (64, 768, 1), 0), out=buf116)
    del permute_155
    buf122 = buf90; del buf90  # reuse
    cpp_fused_view_24(c_void_p(buf116.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = reinterpret_tensor(buf116, (128, 768), (768, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf122, permute_161, out=buf123)
    del permute_161
    buf117 = reinterpret_tensor(buf80, (12, 128, 128), (16384, 128, 1), 0); del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf113, (12, 128, 64), (64, 768, 1), 0), permute_156, out=buf117)
    del permute_156
    buf118 = buf79; del buf79  # reuse
    buf119 = reinterpret_tensor(buf117, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf117  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_25(c_void_p(buf119.data_ptr()), c_void_p(getitem_29.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf118.data_ptr()))
    del alias_10
    del getitem_29
    buf120 = reinterpret_tensor(buf113, (12, 64, 128), (8192, 128, 1), 0); del buf113  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_157, reinterpret_tensor(buf119, (12, 128, 128), (16384, 128, 1), 0), out=buf120)
    del permute_157
    buf126 = buf87; del buf87  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf120, (128, 768), (1, 128), 0), permute_166, out=buf126)
    del permute_166
    buf121 = reinterpret_tensor(buf84, (12, 128, 64), (8192, 64, 1), 0); del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf119, (12, 128, 128), (16384, 128, 1), 0), permute_158, out=buf121)
    del permute_158
    buf129 = buf83; del buf83  # reuse
    cpp_fused_view_26(c_void_p(buf121.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = reinterpret_tensor(buf121, (128, 768), (768, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf129, permute_171, out=buf130)
    del permute_171
    buf115 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf137 = empty((768, ), device='cpu', dtype=torch.float32)
    buf138 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf110.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(mul_21.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    buf124 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (768, 128), (1, 768), 0), view_69, out=buf124)
    buf125 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf122.data_ptr()), c_void_p(buf125.data_ptr()))
    buf127 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf120, (768, 128), (128, 1), 0), view_69, out=buf127)
    buf128 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_29(c_void_p(buf120.data_ptr()), c_void_p(buf128.data_ptr()))
    buf131 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (768, 128), (1, 768), 0), view_69, out=buf131)
    del view_69
    buf132 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf133 = buf110; del buf110  # reuse
    buf134 = buf109; del buf109  # reuse
    buf135 = buf108; del buf108  # reuse
    buf136 = buf133; del buf133  # reuse
    buf139 = reinterpret_tensor(buf120, (1, 128, 768), (98304, 768, 1), 0); del buf120  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_30(c_void_p(buf136.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul_21.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(getitem_25.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf139.data_ptr()))
    del div_24
    del getitem_25
    del mul_21
    del primals_51
    buf140 = reinterpret_tensor(buf104, (128, 3072), (3072, 1), 0); del buf104  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (128, 768), (768, 1), 0), permute_175, out=buf140)
    del permute_175
    buf141 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (768, 128), (1, 768), 0), view_67, out=buf141)
    del view_67
    buf142 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf143 = reinterpret_tensor(buf140, (1, 128, 3072), (393216, 3072, 1), 0); del buf140  # reuse
    cpp_fused_gelu_gelu_backward_sum_31(c_void_p(buf143.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf142.data_ptr()))
    del addmm_16
    buf144 = reinterpret_tensor(buf139, (128, 768), (768, 1), 0); del buf139  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf143, (128, 3072), (3072, 1), 0), permute_179, out=buf144)
    del permute_179
    buf145 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf143, (3072, 128), (1, 3072), 0), view_65, out=buf145)
    del view_65
    buf146 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf147 = buf135; del buf135  # reuse
    buf148 = buf134; del buf134  # reuse
    buf149 = reinterpret_tensor(buf130, (1, 128, 768), (98304, 768, 1), 0); del buf130  # reuse
    buf150 = empty((768, ), device='cpu', dtype=torch.float32)
    buf151 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_32(c_void_p(buf143.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del div_25
    del mul_16
    del primals_45
    buf152 = buf144; del buf144  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (128, 768), (768, 1), 0), permute_183, out=buf152)
    del permute_183
    buf153 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (768, 128), (1, 768), 0), view_63, out=buf153)
    del view_63
    buf155 = reinterpret_tensor(buf136, (12, 128, 64), (8192, 64, 1), 0); del buf136  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_188, reinterpret_tensor(buf152, (12, 128, 64), (64, 768, 1), 0), out=buf155)
    del permute_188
    buf161 = buf129; del buf129  # reuse
    cpp_fused_view_33(c_void_p(buf155.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = reinterpret_tensor(buf155, (128, 768), (768, 1), 0); del buf155  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf161, permute_194, out=buf162)
    del permute_194
    buf156 = reinterpret_tensor(buf119, (12, 128, 128), (16384, 128, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf152, (12, 128, 64), (64, 768, 1), 0), permute_189, out=buf156)
    del permute_189
    buf157 = buf118; del buf118  # reuse
    buf158 = reinterpret_tensor(buf156, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf156  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_34(c_void_p(buf158.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(alias_11.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf157.data_ptr()))
    del alias_11
    del getitem_21
    buf159 = reinterpret_tensor(buf152, (12, 64, 128), (8192, 128, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_190, reinterpret_tensor(buf158, (12, 128, 128), (16384, 128, 1), 0), out=buf159)
    del permute_190
    buf165 = buf126; del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (128, 768), (1, 128), 0), permute_199, out=buf165)
    del permute_199
    buf160 = reinterpret_tensor(buf123, (12, 128, 64), (8192, 64, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf158, (12, 128, 128), (16384, 128, 1), 0), permute_191, out=buf160)
    del permute_191
    buf168 = buf122; del buf122  # reuse
    cpp_fused_view_35(c_void_p(buf160.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = reinterpret_tensor(buf160, (128, 768), (768, 1), 0); del buf160  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf168, permute_204, out=buf169)
    del permute_204
    buf154 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf176 = empty((768, ), device='cpu', dtype=torch.float32)
    buf177 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_36(c_void_p(buf149.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(mul_14.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    buf163 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (768, 128), (1, 768), 0), view_46, out=buf163)
    buf164 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_37(c_void_p(buf161.data_ptr()), c_void_p(buf164.data_ptr()))
    buf166 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (768, 128), (128, 1), 0), view_46, out=buf166)
    buf167 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_38(c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()))
    buf170 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (768, 128), (1, 768), 0), view_46, out=buf170)
    del view_46
    buf171 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf172 = buf149; del buf149  # reuse
    buf173 = buf148; del buf148  # reuse
    buf174 = buf147; del buf147  # reuse
    buf175 = buf172; del buf172  # reuse
    buf178 = reinterpret_tensor(buf159, (1, 128, 768), (98304, 768, 1), 0); del buf159  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_39(c_void_p(buf175.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(mul_14.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf178.data_ptr()))
    del div_27
    del getitem_17
    del mul_14
    del primals_35
    buf179 = reinterpret_tensor(buf143, (128, 3072), (3072, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (128, 768), (768, 1), 0), permute_208, out=buf179)
    del permute_208
    buf180 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (768, 128), (1, 768), 0), view_44, out=buf180)
    del view_44
    buf181 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf182 = reinterpret_tensor(buf179, (1, 128, 3072), (393216, 3072, 1), 0); del buf179  # reuse
    cpp_fused_gelu_gelu_backward_sum_40(c_void_p(buf182.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf181.data_ptr()))
    del addmm_10
    buf183 = reinterpret_tensor(buf178, (128, 768), (768, 1), 0); del buf178  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (128, 3072), (3072, 1), 0), permute_212, out=buf183)
    del permute_212
    buf184 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf182, (3072, 128), (1, 3072), 0), view_42, out=buf184)
    del view_42
    buf185 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf186 = buf174; del buf174  # reuse
    buf187 = buf173; del buf173  # reuse
    buf188 = reinterpret_tensor(buf169, (1, 128, 768), (98304, 768, 1), 0); del buf169  # reuse
    buf189 = empty((768, ), device='cpu', dtype=torch.float32)
    buf190 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_41(c_void_p(buf182.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    del div_28
    del mul_9
    del primals_29
    buf191 = buf183; del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (128, 768), (768, 1), 0), permute_216, out=buf191)
    del permute_216
    buf192 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (768, 128), (1, 768), 0), view_40, out=buf192)
    del view_40
    buf194 = reinterpret_tensor(buf175, (12, 128, 64), (8192, 64, 1), 0); del buf175  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_221, reinterpret_tensor(buf191, (12, 128, 64), (64, 768, 1), 0), out=buf194)
    del permute_221
    buf200 = buf168; del buf168  # reuse
    cpp_fused_view_42(c_void_p(buf194.data_ptr()), c_void_p(buf200.data_ptr()))
    buf201 = reinterpret_tensor(buf194, (128, 768), (768, 1), 0); del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf200, permute_227, out=buf201)
    del permute_227
    buf195 = reinterpret_tensor(buf158, (12, 128, 128), (16384, 128, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf191, (12, 128, 64), (64, 768, 1), 0), permute_222, out=buf195)
    del permute_222
    buf196 = buf157; del buf157  # reuse
    buf197 = reinterpret_tensor(buf195, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf195  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_43(c_void_p(buf197.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(alias_12.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf196.data_ptr()))
    del alias_12
    del getitem_13
    buf198 = reinterpret_tensor(buf191, (12, 64, 128), (8192, 128, 1), 0); del buf191  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_223, reinterpret_tensor(buf197, (12, 128, 128), (16384, 128, 1), 0), out=buf198)
    del permute_223
    buf204 = buf165; del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (128, 768), (1, 128), 0), permute_232, out=buf204)
    del permute_232
    buf199 = reinterpret_tensor(buf162, (12, 128, 64), (8192, 64, 1), 0); del buf162  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf197, (12, 128, 128), (16384, 128, 1), 0), permute_224, out=buf199)
    del permute_224
    buf207 = buf161; del buf161  # reuse
    cpp_fused_view_44(c_void_p(buf199.data_ptr()), c_void_p(buf207.data_ptr()))
    buf208 = reinterpret_tensor(buf199, (128, 768), (768, 1), 0); del buf199  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf207, permute_237, out=buf208)
    del permute_237
    buf193 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf215 = empty((768, ), device='cpu', dtype=torch.float32)
    buf216 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_45(c_void_p(buf188.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(mul_7.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    buf202 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf200, (768, 128), (1, 768), 0), view_23, out=buf202)
    buf203 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_46(c_void_p(buf200.data_ptr()), c_void_p(buf203.data_ptr()))
    buf205 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (768, 128), (128, 1), 0), view_23, out=buf205)
    buf206 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_47(c_void_p(buf198.data_ptr()), c_void_p(buf206.data_ptr()))
    buf209 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (768, 128), (1, 768), 0), view_23, out=buf209)
    del view_23
    buf210 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf211 = buf188; del buf188  # reuse
    buf212 = buf187; del buf187  # reuse
    buf213 = buf186; del buf186  # reuse
    buf214 = buf211; del buf211  # reuse
    buf217 = reinterpret_tensor(buf198, (1, 128, 768), (98304, 768, 1), 0); del buf198  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_48(c_void_p(buf214.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(mul_7.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(getitem_9.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf217.data_ptr()))
    del div_30
    del getitem_9
    del mul_7
    del primals_19
    buf218 = reinterpret_tensor(buf182, (128, 3072), (3072, 1), 0); del buf182  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (128, 768), (768, 1), 0), permute_241, out=buf218)
    del permute_241
    buf219 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (768, 128), (1, 768), 0), view_21, out=buf219)
    del view_21
    buf220 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf221 = reinterpret_tensor(buf218, (1, 128, 3072), (393216, 3072, 1), 0); del buf218  # reuse
    cpp_fused_gelu_gelu_backward_sum_49(c_void_p(buf221.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf220.data_ptr()))
    del addmm_4
    buf222 = reinterpret_tensor(buf217, (128, 768), (768, 1), 0); del buf217  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (128, 3072), (3072, 1), 0), permute_245, out=buf222)
    del permute_245
    buf223 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (3072, 128), (1, 3072), 0), view_19, out=buf223)
    del view_19
    buf224 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf225 = buf213; del buf213  # reuse
    buf226 = buf212; del buf212  # reuse
    buf227 = reinterpret_tensor(buf208, (1, 128, 768), (98304, 768, 1), 0); del buf208  # reuse
    buf228 = empty((768, ), device='cpu', dtype=torch.float32)
    buf229 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_50(c_void_p(buf221.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    del div_31
    del mul_2
    del primals_13
    buf230 = buf222; del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (128, 768), (768, 1), 0), permute_249, out=buf230)
    del permute_249
    buf231 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (768, 128), (1, 768), 0), view_17, out=buf231)
    del view_17
    buf232 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_51(c_void_p(buf227.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf214, (12, 128, 64), (8192, 64, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_254, reinterpret_tensor(buf230, (12, 128, 64), (64, 768, 1), 0), out=buf233)
    del permute_254
    buf234 = reinterpret_tensor(buf197, (12, 128, 128), (16384, 128, 1), 0); del buf197  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf230, (12, 128, 64), (64, 768, 1), 0), permute_255, out=buf234)
    del permute_255
    buf235 = buf196; del buf196  # reuse
    buf236 = reinterpret_tensor(buf234, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf234  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_52(c_void_p(buf236.data_ptr()), c_void_p(getitem_5.data_ptr()), c_void_p(alias_13.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf235.data_ptr()))
    del alias_13
    del buf235
    del getitem_5
    del view_12
    buf237 = reinterpret_tensor(buf230, (12, 64, 128), (8192, 128, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_256, reinterpret_tensor(buf236, (12, 128, 128), (16384, 128, 1), 0), out=buf237)
    del permute_256
    buf238 = reinterpret_tensor(buf207, (12, 128, 64), (8192, 64, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf236, (12, 128, 128), (16384, 128, 1), 0), permute_257, out=buf238)
    del buf236
    del permute_257
    buf239 = buf204; del buf204  # reuse
    cpp_fused_view_53(c_void_p(buf233.data_ptr()), c_void_p(buf239.data_ptr()))
    buf240 = reinterpret_tensor(buf233, (128, 768), (768, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf239, permute_260, out=buf240)
    del permute_260
    buf241 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf239, (768, 128), (1, 768), 0), view, out=buf241)
    buf242 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf239.data_ptr()), c_void_p(buf242.data_ptr()))
    buf243 = buf239; del buf239  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (128, 768), (1, 128), 0), permute_265, out=buf243)
    del permute_265
    buf244 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (768, 128), (128, 1), 0), view, out=buf244)
    buf245 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf246 = buf201; del buf201  # reuse
    cpp_fused_sum_view_55(c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    buf247 = reinterpret_tensor(buf238, (128, 768), (768, 1), 0); del buf238  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf246, permute_270, out=buf247)
    del permute_270
    buf248 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf246, (768, 128), (1, 768), 0), view, out=buf248)
    del view
    buf249 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf250 = buf227; del buf227  # reuse
    buf251 = buf226; del buf226  # reuse
    buf252 = buf225; del buf225  # reuse
    buf257 = reinterpret_tensor(buf237, (1, 128, 768), (98304, 768, 1), 0); del buf237  # reuse
    buf261 = reinterpret_tensor(buf200, (1, 128, 768), (98304, 768, 1), 0); del buf200  # reuse
    buf254 = empty((768, ), device='cpu', dtype=torch.float32)
    buf255 = empty((768, ), device='cpu', dtype=torch.float32)
    buf256 = reinterpret_tensor(buf221, (512, 768), (768, 1), 0); del buf221  # reuse
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_56(c_void_p(buf250.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(slice_2.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del buf240
    del buf243
    del buf246
    del buf247
    del buf250
    del buf251
    del buf252
    del div_33
    del getitem_3
    del mul
    del primals_3
    aten.index_put_(buf256, [slice_2], buf257, True)
    del buf257
    del slice_2
    buf260 = empty((30522, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_57(c_void_p(buf260.data_ptr()))
    aten.index_put_(buf260, [primals_108], buf261, True)
    del buf261
    del primals_108
    return (buf260, buf256, buf254, buf255, reinterpret_tensor(buf248, (768, 768), (768, 1), 0), reinterpret_tensor(buf249, (768, ), (1, ), 0), reinterpret_tensor(buf244, (768, 768), (768, 1), 0), reinterpret_tensor(buf245, (768, ), (1, ), 0), reinterpret_tensor(buf241, (768, 768), (768, 1), 0), reinterpret_tensor(buf242, (768, ), (1, ), 0), reinterpret_tensor(buf231, (768, 768), (768, 1), 0), reinterpret_tensor(buf232, (768, ), (1, ), 0), buf228, buf229, reinterpret_tensor(buf223, (3072, 768), (768, 1), 0), reinterpret_tensor(buf224, (3072, ), (1, ), 0), reinterpret_tensor(buf219, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf220, (768, ), (1, ), 0), buf215, buf216, reinterpret_tensor(buf209, (768, 768), (768, 1), 0), reinterpret_tensor(buf210, (768, ), (1, ), 0), reinterpret_tensor(buf205, (768, 768), (768, 1), 0), reinterpret_tensor(buf206, (768, ), (1, ), 0), reinterpret_tensor(buf202, (768, 768), (768, 1), 0), reinterpret_tensor(buf203, (768, ), (1, ), 0), reinterpret_tensor(buf192, (768, 768), (768, 1), 0), reinterpret_tensor(buf193, (768, ), (1, ), 0), buf189, buf190, reinterpret_tensor(buf184, (3072, 768), (768, 1), 0), reinterpret_tensor(buf185, (3072, ), (1, ), 0), reinterpret_tensor(buf180, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf181, (768, ), (1, ), 0), buf176, buf177, reinterpret_tensor(buf170, (768, 768), (768, 1), 0), reinterpret_tensor(buf171, (768, ), (1, ), 0), reinterpret_tensor(buf166, (768, 768), (768, 1), 0), reinterpret_tensor(buf167, (768, ), (1, ), 0), reinterpret_tensor(buf163, (768, 768), (768, 1), 0), reinterpret_tensor(buf164, (768, ), (1, ), 0), reinterpret_tensor(buf153, (768, 768), (768, 1), 0), reinterpret_tensor(buf154, (768, ), (1, ), 0), buf150, buf151, reinterpret_tensor(buf145, (3072, 768), (768, 1), 0), reinterpret_tensor(buf146, (3072, ), (1, ), 0), reinterpret_tensor(buf141, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf142, (768, ), (1, ), 0), buf137, buf138, reinterpret_tensor(buf131, (768, 768), (768, 1), 0), reinterpret_tensor(buf132, (768, ), (1, ), 0), reinterpret_tensor(buf127, (768, 768), (768, 1), 0), reinterpret_tensor(buf128, (768, ), (1, ), 0), reinterpret_tensor(buf124, (768, 768), (768, 1), 0), reinterpret_tensor(buf125, (768, ), (1, ), 0), reinterpret_tensor(buf114, (768, 768), (768, 1), 0), reinterpret_tensor(buf115, (768, ), (1, ), 0), buf111, buf112, reinterpret_tensor(buf106, (3072, 768), (768, 1), 0), reinterpret_tensor(buf107, (3072, ), (1, ), 0), reinterpret_tensor(buf102, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf103, (768, ), (1, ), 0), buf98, buf99, reinterpret_tensor(buf92, (768, 768), (768, 1), 0), reinterpret_tensor(buf93, (768, ), (1, ), 0), reinterpret_tensor(buf88, (768, 768), (768, 1), 0), reinterpret_tensor(buf89, (768, ), (1, ), 0), reinterpret_tensor(buf85, (768, 768), (768, 1), 0), reinterpret_tensor(buf86, (768, ), (1, ), 0), reinterpret_tensor(buf75, (768, 768), (768, 1), 0), reinterpret_tensor(buf76, (768, ), (1, ), 0), buf72, buf73, reinterpret_tensor(buf67, (3072, 768), (768, 1), 0), reinterpret_tensor(buf68, (3072, ), (1, ), 0), reinterpret_tensor(buf63, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf64, (768, ), (1, ), 0), buf59, buf60, reinterpret_tensor(buf53, (768, 768), (768, 1), 0), reinterpret_tensor(buf54, (768, ), (1, ), 0), reinterpret_tensor(buf49, (768, 768), (768, 1), 0), reinterpret_tensor(buf50, (768, ), (1, ), 0), reinterpret_tensor(buf46, (768, 768), (768, 1), 0), reinterpret_tensor(buf47, (768, ), (1, ), 0), reinterpret_tensor(buf36, (768, 768), (768, 1), 0), reinterpret_tensor(buf37, (768, ), (1, ), 0), buf33, buf34, reinterpret_tensor(buf28, (3072, 768), (768, 1), 0), reinterpret_tensor(buf29, (3072, ), (1, ), 0), reinterpret_tensor(buf24, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf25, (768, ), (1, ), 0), buf20, buf21, reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf16, (768, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf7, (30522, 768), (768, 1), 0), reinterpret_tensor(buf8, (30522, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_109 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    slice_2 = rand_strided((1, 128), (512, 1), device='cpu', dtype=torch.int64)
    mul = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_12 = rand_strided((1, 1, 1, 128), (128, 128, 128, 1), device='cpu', dtype=torch.bool)
    getitem_5 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_17 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_2 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_7 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_40 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_14 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_63 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_21 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_86 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_33 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_28 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_92 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_109 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_30 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_111 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_113 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_35 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_45 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_132 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_37 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_136 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_42 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_138 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_36 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_47 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_21 = rand_strided((128, 30522), (30522, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_68 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_72 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_76 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_80 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_84 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_89 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_90 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_8 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_91 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_92 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_95 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_105 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_109 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_113 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_122 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_9 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_124 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_125 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_128 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_133 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_156 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_10 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_157 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_11 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_221 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_223 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_224 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_232 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_256 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_257 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_265 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 30522), (3906816, 30522, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_103, primals_108, primals_109, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, view_138, addmm_36, mul_47, view_140, sub_21, convert_element_type, permute_68, div_14, permute_72, div_15, permute_76, permute_80, div_16, permute_84, permute_89, permute_90, alias_8, permute_91, permute_92, permute_95, permute_100, permute_105, div_18, permute_109, permute_113, div_19, permute_117, permute_122, permute_123, alias_9, permute_124, permute_125, permute_128, permute_133, permute_138, div_21, permute_142, permute_146, div_22, permute_150, permute_155, permute_156, alias_10, permute_157, permute_158, permute_161, permute_166, permute_171, div_24, permute_175, permute_179, div_25, permute_183, permute_188, permute_189, alias_11, permute_190, permute_191, permute_194, permute_199, permute_204, div_27, permute_208, permute_212, div_28, permute_216, permute_221, permute_222, alias_12, permute_223, permute_224, permute_227, permute_232, permute_237, div_30, permute_241, permute_245, div_31, permute_249, permute_254, permute_255, alias_13, permute_256, permute_257, permute_260, permute_265, permute_270, div_33, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForMaskedLM', benchmark_compiled_module)
