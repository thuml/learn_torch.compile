
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


cpp_fused_add_gelu_gelu_backward_mul_sigmoid_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp18 = in_ptr5[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr9[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp18 = in_ptr12[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp30 = in_ptr16[static_cast<long>(0L)];
                        auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp39 = in_ptr19[static_cast<long>(0L)];
                        auto tmp45 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp46 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp50 = in_ptr22[static_cast<long>(0L)];
                        auto tmp54 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp55 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp59 = in_ptr25[static_cast<long>(0L)];
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        auto tmp27 = decltype(tmp26)(1)/(decltype(tmp26)(1) + tmp26.neg().exp());
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = tmp28 * tmp5;
                        auto tmp31 = at::vec::Vectorized<float>(tmp30);
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp32 * tmp11;
                        auto tmp36 = decltype(tmp35)(1)/(decltype(tmp35)(1) + tmp35.neg().exp());
                        auto tmp37 = tmp34 * tmp36;
                        auto tmp38 = tmp37 * tmp5;
                        auto tmp40 = at::vec::Vectorized<float>(tmp39);
                        auto tmp41 = tmp38 * tmp40;
                        auto tmp42 = tmp41 * tmp11;
                        auto tmp43 = tmp42 + tmp24;
                        auto tmp44 = tmp33 + tmp43;
                        auto tmp47 = decltype(tmp46)(1)/(decltype(tmp46)(1) + tmp46.neg().exp());
                        auto tmp48 = tmp45 * tmp47;
                        auto tmp49 = tmp48 * tmp5;
                        auto tmp51 = at::vec::Vectorized<float>(tmp50);
                        auto tmp52 = tmp49 * tmp51;
                        auto tmp53 = tmp52 * tmp11;
                        auto tmp56 = decltype(tmp55)(1)/(decltype(tmp55)(1) + tmp55.neg().exp());
                        auto tmp57 = tmp54 * tmp56;
                        auto tmp58 = tmp57 * tmp5;
                        auto tmp60 = at::vec::Vectorized<float>(tmp59);
                        auto tmp61 = tmp58 * tmp60;
                        auto tmp62 = tmp61 * tmp11;
                        auto tmp63 = tmp62 + tmp44;
                        auto tmp64 = tmp53 + tmp63;
                        auto tmp65 = static_cast<float>(0.7071067811865476);
                        auto tmp66 = at::vec::Vectorized<float>(tmp65);
                        auto tmp67 = tmp63 * tmp66;
                        auto tmp68 = tmp67.erf();
                        auto tmp69 = static_cast<float>(1.0);
                        auto tmp70 = at::vec::Vectorized<float>(tmp69);
                        auto tmp71 = tmp68 + tmp70;
                        auto tmp72 = static_cast<float>(0.5);
                        auto tmp73 = at::vec::Vectorized<float>(tmp72);
                        auto tmp74 = tmp71 * tmp73;
                        auto tmp75 = tmp63 * tmp63;
                        auto tmp76 = static_cast<float>(-0.5);
                        auto tmp77 = at::vec::Vectorized<float>(tmp76);
                        auto tmp78 = tmp75 * tmp77;
                        auto tmp79 = tmp78.exp();
                        auto tmp80 = static_cast<float>(0.3989422804014327);
                        auto tmp81 = at::vec::Vectorized<float>(tmp80);
                        auto tmp82 = tmp79 * tmp81;
                        auto tmp83 = tmp63 * tmp82;
                        auto tmp84 = tmp74 + tmp83;
                        auto tmp85 = tmp43 * tmp66;
                        auto tmp86 = tmp85.erf();
                        auto tmp87 = tmp86 + tmp70;
                        auto tmp88 = tmp87 * tmp73;
                        auto tmp89 = tmp43 * tmp43;
                        auto tmp90 = tmp89 * tmp77;
                        auto tmp91 = tmp90.exp();
                        auto tmp92 = tmp91 * tmp81;
                        auto tmp93 = tmp43 * tmp92;
                        auto tmp94 = tmp88 + tmp93;
                        tmp24.store(out_ptr1 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        tmp44.store(out_ptr2 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        tmp64.store(out_ptr3 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        tmp84.store(out_ptr4 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        tmp94.store(out_ptr5 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr28[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp18 = in_ptr31[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        tmp24.store(out_ptr6 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_div_gelu_gelu_backward_mul_sum_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (3072L*x2) + (110592L*x0)), static_cast<long>(3072L), tmp5, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (3072L*x2) + (110592L*x0)), static_cast<long>(3072L), tmp5, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (3072L*x2) + (110592L*x0)), static_cast<long>(3072L), tmp5, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (3072L*x2) + (110592L*x0)), static_cast<long>(3072L), tmp5, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x1 + x1_inner + (3072L*x0))];
                            auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                            auto tmp1 = static_cast<float>(36.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1.7015043497085571);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp7 = static_cast<float>(0.7071067811865476);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp10 = tmp9.erf();
                            auto tmp11 = static_cast<float>(1.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 + tmp12;
                            auto tmp14 = static_cast<float>(0.5);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp17 = tmp6 * tmp6;
                            auto tmp18 = static_cast<float>(-0.5);
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp20.exp();
                            auto tmp22 = static_cast<float>(0.3989422804014327);
                            auto tmp23 = at::vec::Vectorized<float>(tmp22);
                            auto tmp24 = tmp21 * tmp23;
                            auto tmp25 = tmp6 * tmp24;
                            auto tmp26 = tmp16 + tmp25;
                            auto tmp27 = at::vec::Vectorized<float>(tmp4);
                            auto tmp28 = tmp27 * tmp26;
                            tmp28.store(out_ptr1 + static_cast<long>(x2 + (36L*x1) + (36L*x1_inner) + (110592L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(32L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (3072L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (3072L*x2) + (110592L*x0)));
                        auto tmp1 = static_cast<float>(36.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp4 = static_cast<float>(1.7015043497085571);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = static_cast<float>(0.7071067811865476);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp10.erf();
                        auto tmp12 = static_cast<float>(1.0);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 + tmp13;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp7 * tmp7;
                        auto tmp19 = static_cast<float>(-0.5);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 * tmp20;
                        auto tmp22 = tmp21.exp();
                        auto tmp23 = static_cast<float>(0.3989422804014327);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 * tmp24;
                        auto tmp26 = tmp7 * tmp25;
                        auto tmp27 = tmp17 + tmp26;
                        auto tmp28 = tmp6 * tmp27;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp28.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (36L*x1) + (36L*x1_inner) + (110592L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec) 
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr3[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (55296L*x0)));
                            auto tmp4 = in_ptr8[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x2) + (55296L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(36.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr2;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9622504486493761);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = static_cast<float>(1.7015043497085571);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = static_cast<float>(0.7071067811865476);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp11.erf();
                auto tmp13 = static_cast<float>(1.0);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 + tmp14;
                auto tmp16 = static_cast<float>(0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp8 * tmp8;
                auto tmp20 = static_cast<float>(-0.5);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp22.exp();
                auto tmp24 = static_cast<float>(0.3989422804014327);
                auto tmp25 = at::vec::Vectorized<float>(tmp24);
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp8 * tmp26;
                auto tmp28 = tmp18 + tmp27;
                auto tmp29 = tmp7 * tmp28;
                auto tmp30 = tmp0 + tmp29;
                tmp30.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec) 
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr3[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (55296L*x0)));
                            auto tmp4 = in_ptr9[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1536L*x2) + (55296L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(36.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr5 = in_out_ptr2;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr7[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.7071067811865476);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp17.erf();
                        auto tmp19 = static_cast<float>(1.0);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = static_cast<float>(0.5);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp25 = tmp14 * tmp14;
                        auto tmp26 = static_cast<float>(-0.5);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = tmp28.exp();
                        auto tmp30 = static_cast<float>(0.3989422804014327);
                        auto tmp31 = at::vec::Vectorized<float>(tmp30);
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp14 * tmp32;
                        auto tmp34 = tmp24 + tmp33;
                        tmp34.store(out_ptr3 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9805806756909201);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = static_cast<float>(1.7015043497085571);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp0 + tmp9;
                tmp10.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec) 
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (55296L*x0)));
                            auto tmp4 = in_ptr7[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (55296L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(36.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (55296L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(110592L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1.7015043497085571);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.7071067811865476);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp7.erf();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp4 * tmp4;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp18.exp();
                    auto tmp20 = static_cast<float>(0.3989422804014327);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = tmp4 * tmp22;
                    auto tmp24 = tmp14 + tmp23;
                    auto tmp25 = tmp3 * tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_19 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x3 + (768L*x2) + (9216L*x1) + (110592L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(13);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x3 + (768L*x2) + (9984L*x1) + (129792L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(1.7015043497085571);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp13 = static_cast<float>(0.7071067811865476);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp15.erf();
                            auto tmp17 = static_cast<float>(1.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = static_cast<float>(0.5);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp23 = tmp12 * tmp12;
                            auto tmp24 = static_cast<float>(-0.5);
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp27 = tmp26.exp();
                            auto tmp28 = static_cast<float>(0.3989422804014327);
                            auto tmp29 = at::vec::Vectorized<float>(tmp28);
                            auto tmp30 = tmp27 * tmp29;
                            auto tmp31 = tmp12 * tmp30;
                            auto tmp32 = tmp22 + tmp31;
                            auto tmp33 = tmp11 * tmp32;
                            tmp33.store(out_ptr4 + static_cast<long>(x3 + (768L*x2) + (9216L*x1) + (110592L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1536L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr1[static_cast<long>(x3 + (1536L*x2) + (18432L*x1) + (221184L*x0))];
                            auto tmp1 = in_ptr5[static_cast<long>(x3 + (1536L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(6L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (1536L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(6L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 6L)) + (9216L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(6L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (9216L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(6L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 6L)) + (55296L*x0))];
                            auto tmp17 = in_ptr6[static_cast<long>(x3 + (1536L*x2) + (18432L*x1) + (221184L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(6L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(6L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.8980265101338745);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = static_cast<float>(1.7015043497085571);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp18 = static_cast<float>(0.7071067811865476);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = std::erf(tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            auto tmp23 = static_cast<float>(0.5);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp17)(tmp17 * tmp17);
                            auto tmp26 = static_cast<float>(-0.5);
                            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                            auto tmp28 = std::exp(tmp27);
                            auto tmp29 = static_cast<float>(0.3989422804014327);
                            auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                            auto tmp31 = decltype(tmp17)(tmp17 * tmp30);
                            auto tmp32 = decltype(tmp24)(tmp24 + tmp31);
                            auto tmp33 = decltype(tmp16)(tmp16 * tmp32);
                            in_out_ptr1[static_cast<long>(x3 + (1536L*x2) + (18432L*x1) + (221184L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp4 = in_ptr7[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(144.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_gelu_backward_mul_sigmoid_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            float tmp_acc1 = 0;
            at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc1) reduction(+:tmp_acc0_vec) reduction(+:tmp_acc1_vec)  collapse(2)
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.9128709291752768);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = static_cast<float>(1.7015043497085571);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 * tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = tmp0 + tmp20;
                            auto tmp22 = tmp21 * tmp2;
                            auto tmp25 = decltype(tmp24)(1)/(decltype(tmp24)(1) + tmp24.neg().exp());
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp27 = tmp26 * tmp9;
                            auto tmp28 = tmp22 * tmp27;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp28;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr0[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
            tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
            out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc1);
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp14 = in_ptr8[static_cast<long>(0L)];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp2 = static_cast<float>(0.9128709291752768);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(1.7015043497085571);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp10 = tmp0 + tmp9;
                            auto tmp11 = static_cast<float>(0.2);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp17 = static_cast<float>(2.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_gelu_backward_mul_sigmoid_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp14 = in_ptr3[static_cast<long>(0L)];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp2 = static_cast<float>(0.9128709291752768);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = static_cast<float>(1.7015043497085571);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = static_cast<float>(0.2);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = static_cast<float>(2.0);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = decltype(tmp20)(1)/(decltype(tmp20)(1) + tmp20.neg().exp());
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp24 = static_cast<float>(144.0);
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp23 / tmp25;
                        auto tmp27 = tmp22 + tmp26;
                        tmp27.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(884736L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9128709291752768);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = static_cast<float>(1.7015043497085571);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp0 + tmp9;
                auto tmp12 = static_cast<float>(0.9284766908852592);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp14 * tmp6;
                auto tmp17 = static_cast<float>(0.7071067811865476);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp19.erf();
                auto tmp21 = static_cast<float>(1.0);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 + tmp22;
                auto tmp24 = static_cast<float>(0.5);
                auto tmp25 = at::vec::Vectorized<float>(tmp24);
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp16 * tmp16;
                auto tmp28 = static_cast<float>(-0.5);
                auto tmp29 = at::vec::Vectorized<float>(tmp28);
                auto tmp30 = tmp27 * tmp29;
                auto tmp31 = tmp30.exp();
                auto tmp32 = static_cast<float>(0.3989422804014327);
                auto tmp33 = at::vec::Vectorized<float>(tmp32);
                auto tmp34 = tmp31 * tmp33;
                auto tmp35 = tmp16 * tmp34;
                auto tmp36 = tmp26 + tmp35;
                auto tmp37 = tmp15 * tmp36;
                auto tmp38 = tmp10 + tmp37;
                tmp38.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp4 = in_ptr9[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(144.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_gelu_backward_mul_sigmoid_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            float tmp_acc1 = 0;
            at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc1) reduction(+:tmp_acc0_vec) reduction(+:tmp_acc1_vec)  collapse(2)
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = static_cast<float>(0.9449111825230679);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = static_cast<float>(1.7015043497085571);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 * tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = tmp0 + tmp20;
                            auto tmp22 = tmp21 * tmp2;
                            auto tmp25 = decltype(tmp24)(1)/(decltype(tmp24)(1) + tmp24.neg().exp());
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp27 = tmp26 * tmp9;
                            auto tmp28 = tmp22 * tmp27;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp28;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr0[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
            tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
            out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc1);
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp14 = in_ptr8[static_cast<long>(0L)];
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp2 = static_cast<float>(0.9449111825230679);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = static_cast<float>(1.7015043497085571);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp10 = tmp0 + tmp9;
                            auto tmp11 = static_cast<float>(0.2);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp17 = static_cast<float>(2.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp21 = tmp19 * tmp20;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_gelu_backward_mul_sigmoid_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp14 = in_ptr3[static_cast<long>(0L)];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp2 = static_cast<float>(0.9449111825230679);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = static_cast<float>(1.7015043497085571);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = static_cast<float>(0.2);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = static_cast<float>(2.0);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = decltype(tmp20)(1)/(decltype(tmp20)(1) + tmp20.neg().exp());
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp24 = static_cast<float>(144.0);
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp23 / tmp25;
                        auto tmp27 = tmp22 + tmp26;
                        tmp27.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr2;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(884736L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9449111825230679);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = static_cast<float>(1.7015043497085571);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp0 + tmp9;
                auto tmp12 = static_cast<float>(0.9622504486493761);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp14 * tmp6;
                auto tmp17 = static_cast<float>(0.7071067811865476);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp19.erf();
                auto tmp21 = static_cast<float>(1.0);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 + tmp22;
                auto tmp24 = static_cast<float>(0.5);
                auto tmp25 = at::vec::Vectorized<float>(tmp24);
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp16 * tmp16;
                auto tmp28 = static_cast<float>(-0.5);
                auto tmp29 = at::vec::Vectorized<float>(tmp28);
                auto tmp30 = tmp27 * tmp29;
                auto tmp31 = tmp30.exp();
                auto tmp32 = static_cast<float>(0.3989422804014327);
                auto tmp33 = at::vec::Vectorized<float>(tmp32);
                auto tmp34 = tmp31 * tmp33;
                auto tmp35 = tmp16 * tmp34;
                auto tmp36 = tmp26 + tmp35;
                auto tmp37 = tmp15 * tmp36;
                auto tmp38 = tmp10 + tmp37;
                tmp38.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec)  collapse(2)
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr3[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp4 = in_ptr11[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(144.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr5 = in_out_ptr2;
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02551551815399144);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.02551551815399144);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr7[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.7071067811865476);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp17.erf();
                        auto tmp19 = static_cast<float>(1.0);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = static_cast<float>(0.5);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp25 = tmp14 * tmp14;
                        auto tmp26 = static_cast<float>(-0.5);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = tmp28.exp();
                        auto tmp30 = static_cast<float>(0.3989422804014327);
                        auto tmp31 = at::vec::Vectorized<float>(tmp30);
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp14 * tmp32;
                        auto tmp34 = tmp24 + tmp33;
                        tmp34.store(out_ptr3 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(884736L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9805806756909201);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = static_cast<float>(1.7015043497085571);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp0 + tmp9;
                tmp10.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec)  collapse(2)
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (1536L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp4 = in_ptr7[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (221184L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(144.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (221184L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
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
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.03608439182435161);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0013020833333333333);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.03608439182435161);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(442368L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_64 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x3 + (768L*x2) + (18432L*x1) + (442368L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(25);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x3 + (768L*x2) + (19200L*x1) + (480000L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(1.7015043497085571);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp13 = static_cast<float>(0.7071067811865476);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp15.erf();
                            auto tmp17 = static_cast<float>(1.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = static_cast<float>(0.5);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp23 = tmp12 * tmp12;
                            auto tmp24 = static_cast<float>(-0.5);
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp27 = tmp26.exp();
                            auto tmp28 = static_cast<float>(0.3989422804014327);
                            auto tmp29 = at::vec::Vectorized<float>(tmp28);
                            auto tmp30 = tmp27 * tmp29;
                            auto tmp31 = tmp12 * tmp30;
                            auto tmp32 = tmp22 + tmp31;
                            auto tmp33 = tmp11 * tmp32;
                            tmp33.store(out_ptr4 + static_cast<long>(x3 + (768L*x2) + (18432L*x1) + (442368L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04419417382415922);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.001953125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04419417382415922);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04419417382415922);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.001953125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04419417382415922);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x3 + (512L*x2) + (12288L*x1) + (294912L*x0))];
                            auto tmp1 = in_ptr6[static_cast<long>(x3 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(12L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(12L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 12L)) + (6144L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(12L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (6144L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(12L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 12L)) + (73728L*x0))];
                            auto tmp17 = in_out_ptr1[static_cast<long>(x3 + (512L*x2) + (12288L*x1) + (294912L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(12L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(12L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.9622504486493761);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = static_cast<float>(1.7015043497085571);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp18 = static_cast<float>(0.7071067811865476);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = std::erf(tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            auto tmp23 = static_cast<float>(0.5);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp17)(tmp17 * tmp17);
                            auto tmp26 = static_cast<float>(-0.5);
                            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                            auto tmp28 = std::exp(tmp27);
                            auto tmp29 = static_cast<float>(0.3989422804014327);
                            auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                            auto tmp31 = decltype(tmp17)(tmp17 * tmp30);
                            auto tmp32 = decltype(tmp24)(tmp24 + tmp31);
                            auto tmp33 = decltype(tmp16)(tmp16 * tmp32);
                            in_out_ptr1[static_cast<long>(x3 + (512L*x2) + (12288L*x1) + (294912L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec)  collapse(2)
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (512L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr3[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x2) + (294912L*x0)));
                            auto tmp4 = in_ptr9[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x2) + (294912L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (294912L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(576.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (294912L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.0625);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.00390625);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.0625);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr5 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04419417382415922);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.001953125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04419417382415922);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp7 = in_ptr7[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.7071067811865476);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp17.erf();
                        auto tmp19 = static_cast<float>(1.0);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = static_cast<float>(0.5);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp25 = tmp14 * tmp14;
                        auto tmp26 = static_cast<float>(-0.5);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = tmp28.exp();
                        auto tmp30 = static_cast<float>(0.3989422804014327);
                        auto tmp31 = at::vec::Vectorized<float>(tmp30);
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp14 * tmp32;
                        auto tmp34 = tmp24 + tmp33;
                        tmp34.store(out_ptr3 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9805806756909201);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = static_cast<float>(1.7015043497085571);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp0 + tmp9;
                tmp10.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec)  collapse(2)
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (512L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x2) + (294912L*x0)));
                            auto tmp4 = in_ptr7[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x2) + (294912L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(576.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (294912L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.0625);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.00390625);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.0625);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_77 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x3 + (256L*x2) + (12288L*x1) + (589824L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(49);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x3 + (256L*x2) + (12544L*x1) + (614656L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(1.7015043497085571);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp13 = static_cast<float>(0.7071067811865476);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp15.erf();
                            auto tmp17 = static_cast<float>(1.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = static_cast<float>(0.5);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp23 = tmp12 * tmp12;
                            auto tmp24 = static_cast<float>(-0.5);
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp27 = tmp26.exp();
                            auto tmp28 = static_cast<float>(0.3989422804014327);
                            auto tmp29 = at::vec::Vectorized<float>(tmp28);
                            auto tmp30 = tmp27 * tmp29;
                            auto tmp31 = tmp12 * tmp30;
                            auto tmp32 = tmp22 + tmp31;
                            auto tmp33 = tmp11 * tmp32;
                            tmp33.store(out_ptr4 + static_cast<long>(x3 + (256L*x2) + (12288L*x1) + (589824L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_78 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.0625);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.00390625);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.0625);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr5 = in_out_ptr2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.0625);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.00390625);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.0625);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (256L*x1) + (589824L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp7 = in_ptr7[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (256L*x1) + (589824L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.7071067811865476);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp17.erf();
                        auto tmp19 = static_cast<float>(1.0);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = static_cast<float>(0.5);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp25 = tmp14 * tmp14;
                        auto tmp26 = static_cast<float>(-0.5);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = tmp28.exp();
                        auto tmp30 = static_cast<float>(0.3989422804014327);
                        auto tmp31 = at::vec::Vectorized<float>(tmp30);
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp14 * tmp32;
                        auto tmp34 = tmp24 + tmp33;
                        tmp34.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (589824L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr1[static_cast<long>(x3 + (256L*x2) + (12288L*x1) + (589824L*x0))];
                            auto tmp1 = in_ptr9[static_cast<long>(x3 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(24L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(24L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 24L)) + (6144L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(24L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (6144L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(24L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 24L)) + (147456L*x0))];
                            auto tmp17 = out_ptr3[static_cast<long>(x3 + (256L*x2) + (12288L*x1) + (589824L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(24L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(24L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.9805806756909201);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp15 = static_cast<float>(1.7015043497085571);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            in_out_ptr1[static_cast<long>(x3 + (256L*x2) + (12288L*x1) + (589824L*x0))] = tmp18;
                        }
                    }
                }
            }
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp parallel num_threads(28)
            {
                #pragma omp for reduction(+:tmp_acc0) reduction(+:tmp_acc0_vec)  collapse(2)
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (256L*x1) + (589824L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (256L*x1) + (589824L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (256L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = static_cast<float>(2.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x2) + (589824L*x0)));
                            auto tmp4 = in_ptr7[static_cast<long>(0L)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x2) + (589824L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp7 = static_cast<float>(2.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            auto tmp11 = tmp9 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_sigmoid_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (589824L*x0)));
                        auto tmp4 = in_ptr1[static_cast<long>(0L)];
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = static_cast<float>(2.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = decltype(tmp10)(1)/(decltype(tmp10)(1) + tmp10.neg().exp());
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(2304.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = tmp12 + tmp16;
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (589824L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.08838834764831845);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.0078125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.08838834764831845);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((128L*(static_cast<long>(x1) % static_cast<long>(9L))) + (1152L*x0) + (1152L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.02946278254943948);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (128L*x1) + (1152L*x0) + (1152L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.0008680555555555555);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.02946278254943948);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (1152L*x0) + (1152L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_85 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.08838834764831845);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.0078125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.08838834764831845);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.08838834764831845);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.0078125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.08838834764831845);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = static_cast<float>(1.7015043497085571);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp10 = static_cast<float>(0.7071067811865476);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp12.erf();
                auto tmp14 = tmp13 + tmp4;
                auto tmp15 = static_cast<float>(0.5);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp9 * tmp9;
                auto tmp19 = static_cast<float>(-0.5);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp21.exp();
                auto tmp23 = static_cast<float>(0.3989422804014327);
                auto tmp24 = at::vec::Vectorized<float>(tmp23);
                auto tmp25 = tmp22 * tmp24;
                auto tmp26 = tmp9 * tmp25;
                auto tmp27 = tmp17 + tmp26;
                auto tmp28 = tmp8 * tmp27;
                tmp28.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_87 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            auto tmp3 = static_cast<float>(0.041666666666666664);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.041666666666666664);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x3 + (64L*x2) + (6144L*x1) + (589824L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(97);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x3 + (64L*x2) + (6208L*x1) + (602176L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(1.7015043497085571);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp13 = static_cast<float>(0.7071067811865476);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp15.erf();
                            auto tmp17 = static_cast<float>(1.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            auto tmp20 = static_cast<float>(0.5);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp23 = tmp12 * tmp12;
                            auto tmp24 = static_cast<float>(-0.5);
                            auto tmp25 = at::vec::Vectorized<float>(tmp24);
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp27 = tmp26.exp();
                            auto tmp28 = static_cast<float>(0.3989422804014327);
                            auto tmp29 = at::vec::Vectorized<float>(tmp28);
                            auto tmp30 = tmp27 * tmp29;
                            auto tmp31 = tmp12 * tmp30;
                            auto tmp32 = tmp22 + tmp31;
                            auto tmp33 = tmp11 * tmp32;
                            tmp33.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (6144L*x1) + (589824L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_88 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x1) % static_cast<long>(9L))) + (288L*x0) + (288L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((32L*(static_cast<long>(x1) % static_cast<long>(9L))) + (288L*x0) + (288L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.05892556509887896);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (32L*x1) + (288L*x0) + (288L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (32L*x1) + (288L*x0) + (288L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.003472222222222222);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.05892556509887896);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_89 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(9L))) + (144L*x0) + (144L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(9L))) + (144L*x0) + (144L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.08333333333333333);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (16L*x1) + (144L*x0) + (144L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (16L*x1) + (144L*x0) + (144L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.006944444444444444);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.08333333333333333);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (144L*x0) + (144L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(1.7015043497085571);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp5 = static_cast<float>(0.7071067811865476);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = tmp7.erf();
                auto tmp9 = static_cast<float>(1.0);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = static_cast<float>(0.5);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp4 * tmp4;
                auto tmp16 = static_cast<float>(-0.5);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp15 * tmp17;
                auto tmp19 = tmp18.exp();
                auto tmp20 = static_cast<float>(0.3989422804014327);
                auto tmp21 = at::vec::Vectorized<float>(tmp20);
                auto tmp22 = tmp19 * tmp21;
                auto tmp23 = tmp4 * tmp22;
                auto tmp24 = tmp14 + tmp23;
                auto tmp25 = tmp3 * tmp24;
                tmp25.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_view_90 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((3L*(static_cast<long>(x1) % static_cast<long>(9L))) + (27L*x0) + (27L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((3L*(static_cast<long>(x1) % static_cast<long>(9L))) + (27L*x0) + (27L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.19245008972987526);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (3L*x1) + (27L*x0) + (27L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (3L*x1) + (27L*x0) + (27L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.037037037037037035);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.19245008972987526);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (27L*x0) + (27L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_83, primals_84, primals_86, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_156, primals_158, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, constant_pad_nd, squeeze_1, view_2, convolution, mul_6, squeeze_3, view_5, convolution_1, mul_13, squeeze_5, view_8, convolution_2, constant_pad_nd_1, squeeze_7, view_11, convolution_3, mul_28, squeeze_9, view_14, convolution_4, squeeze_11, view_17, convolution_5, mul_38, squeeze_13, view_20, convolution_6, mul_45, squeeze_15, view_23, convolution_7, mul_52, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_64, avg_pool2d, squeeze_19, view_29, convolution_11, squeeze_21, view_32, convolution_12, constant_pad_nd_2, squeeze_23, view_35, convolution_13, mul_81, squeeze_25, view_38, convolution_14, mul_88, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_100, squeeze_29, view_44, convolution_18, mul_107, squeeze_31, view_47, convolution_19, mul_114, squeeze_33, view_50, convolution_20, mul_121, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_133, avg_pool2d_1, squeeze_37, view_56, convolution_24, squeeze_39, view_59, convolution_25, constant_pad_nd_3, squeeze_41, view_62, convolution_26, mul_150, squeeze_43, view_65, convolution_27, mul_157, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_169, squeeze_47, view_71, convolution_31, mul_176, squeeze_49, view_74, convolution_32, mul_183, squeeze_51, view_77, convolution_33, mul_190, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_202, squeeze_55, view_83, convolution_37, mul_209, squeeze_57, view_86, convolution_38, mul_216, squeeze_59, view_89, convolution_39, mul_223, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_235, squeeze_63, view_95, convolution_43, mul_242, squeeze_65, view_98, convolution_44, mul_249, squeeze_67, view_101, convolution_45, mul_256, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_268, squeeze_71, view_107, convolution_49, mul_275, squeeze_73, view_110, convolution_50, mul_282, squeeze_75, view_113, convolution_51, mul_289, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_301, squeeze_79, view_119, convolution_55, mul_308, squeeze_81, view_122, convolution_56, mul_315, squeeze_83, view_125, convolution_57, mul_322, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_334, avg_pool2d_2, squeeze_87, view_131, convolution_61, squeeze_89, view_134, convolution_62, constant_pad_nd_4, squeeze_91, view_137, convolution_63, mul_351, squeeze_93, view_140, convolution_64, mul_358, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_370, squeeze_97, view_146, convolution_68, mul_377, squeeze_99, view_149, convolution_69, mul_384, squeeze_101, view_152, convolution_70, mul_391, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_403, squeeze_105, view_158, convolution_74, mul_410, squeeze_107, view_161, convolution_75, mul_417, squeeze_109, view_164, convolution_76, mul_424, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_118, squeeze_113, view_170, convolution_80, clone_12, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_4, (32, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_5, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_8, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_10, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_11, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_13, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_14, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_16, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_19, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_20, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_22, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_23, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_25, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_26, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_28, (), ())
    assert_size_stride(primals_29, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_30, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_32, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_33, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_35, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_36, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_38, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_39, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_41, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_42, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_44, (), ())
    assert_size_stride(primals_45, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_46, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_48, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_49, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_51, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_52, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_54, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_55, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_57, (), ())
    assert_size_stride(primals_58, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_61, (768, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_62, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_64, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_65, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_67, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_68, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_70, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_71, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_73, (), ())
    assert_size_stride(primals_74, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_75, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_77, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_78, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_80, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_81, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_83, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_84, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_86, (), ())
    assert_size_stride(primals_87, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_88, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_90, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_91, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_93, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_94, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_96, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_97, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_99, (), ())
    assert_size_stride(primals_100, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_101, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_103, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_104, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_106, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_107, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_109, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_110, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_112, (), ())
    assert_size_stride(primals_113, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_114, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_116, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_117, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_119, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_120, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_122, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_123, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_125, (), ())
    assert_size_stride(primals_126, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_127, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_129, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_130, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_132, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_133, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_135, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_136, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_138, (), ())
    assert_size_stride(primals_139, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_140, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_142, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_143, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_145, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_146, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_148, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_149, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_151, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_152, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_154, (), ())
    assert_size_stride(primals_155, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_156, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_158, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_159, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_161, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_162, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_164, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_165, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_169, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_171, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_172, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_174, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_175, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_177, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_178, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (3072, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_182, (3072, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_184, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_186, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_188, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_190, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_192, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_194, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_196, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_198, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_200, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_202, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_204, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_206, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_208, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_210, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_212, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_214, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_216, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_218, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_220, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_222, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_224, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_226, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_228, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_230, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(constant_pad_nd, (4, 3, 193, 193), (111747, 1, 579, 3))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(view_2, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(convolution, (4, 16, 96, 96), (147456, 1, 1536, 16))
    assert_size_stride(mul_6, (4, 16, 96, 96), (147456, 1, 1536, 16))
    assert_size_stride(squeeze_3, (32, ), (1, ))
    assert_size_stride(view_5, (32, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(convolution_1, (4, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(mul_13, (4, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(squeeze_5, (64, ), (1, ))
    assert_size_stride(view_8, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(convolution_2, (4, 64, 96, 96), (589824, 1, 6144, 64))
    assert_size_stride(constant_pad_nd_1, (4, 64, 97, 97), (602176, 1, 6208, 64))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(view_11, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_3, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(mul_28, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(squeeze_9, (256, ), (1, ))
    assert_size_stride(view_14, (256, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_4, (4, 256, 48, 48), (589824, 1, 12288, 256))
    assert_size_stride(squeeze_11, (128, ), (1, ))
    assert_size_stride(view_17, (128, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_5, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(mul_38, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(squeeze_13, (128, ), (1, ))
    assert_size_stride(view_20, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_6, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(mul_45, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(squeeze_15, (128, ), (1, ))
    assert_size_stride(view_23, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_7, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(mul_52, (4, 128, 48, 48), (294912, 1, 6144, 128))
    assert_size_stride(squeeze_17, (256, ), (1, ))
    assert_size_stride(view_26, (256, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_8, (4, 256, 48, 48), (589824, 1, 12288, 256))
    assert_size_stride(mean, (4, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu, (4, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_10, (4, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_64, (4, 256, 48, 48), (589824, 1, 12288, 256))
    assert_size_stride(avg_pool2d, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(squeeze_19, (512, ), (1, ))
    assert_size_stride(view_29, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_11, (4, 512, 24, 24), (294912, 1, 12288, 512))
    assert_size_stride(squeeze_21, (256, ), (1, ))
    assert_size_stride(view_32, (256, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_12, (4, 256, 48, 48), (589824, 1, 12288, 256))
    assert_size_stride(constant_pad_nd_2, (4, 256, 49, 49), (614656, 1, 12544, 256))
    assert_size_stride(squeeze_23, (256, ), (1, ))
    assert_size_stride(view_35, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_13, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(mul_81, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(squeeze_25, (256, ), (1, ))
    assert_size_stride(view_38, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_14, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(mul_88, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(squeeze_27, (512, ), (1, ))
    assert_size_stride(view_41, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_15, (4, 512, 24, 24), (294912, 1, 12288, 512))
    assert_size_stride(mean_1, (4, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(relu_1, (4, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_17, (4, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(mul_100, (4, 512, 24, 24), (294912, 1, 12288, 512))
    assert_size_stride(squeeze_29, (256, ), (1, ))
    assert_size_stride(view_44, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_18, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(mul_107, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(view_47, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_19, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(mul_114, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(squeeze_33, (256, ), (1, ))
    assert_size_stride(view_50, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_20, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(mul_121, (4, 256, 24, 24), (147456, 1, 6144, 256))
    assert_size_stride(squeeze_35, (512, ), (1, ))
    assert_size_stride(view_53, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_21, (4, 512, 24, 24), (294912, 1, 12288, 512))
    assert_size_stride(mean_2, (4, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(relu_2, (4, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_23, (4, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(mul_133, (4, 512, 24, 24), (294912, 1, 12288, 512))
    assert_size_stride(avg_pool2d_1, (4, 512, 12, 12), (73728, 1, 6144, 512))
    assert_size_stride(squeeze_37, (1536, ), (1, ))
    assert_size_stride(view_56, (1536, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_24, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(squeeze_39, (768, ), (1, ))
    assert_size_stride(view_59, (768, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_25, (4, 768, 24, 24), (442368, 1, 18432, 768))
    assert_size_stride(constant_pad_nd_3, (4, 768, 25, 25), (480000, 1, 19200, 768))
    assert_size_stride(squeeze_41, (768, ), (1, ))
    assert_size_stride(view_62, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_26, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_150, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_43, (768, ), (1, ))
    assert_size_stride(view_65, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_27, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_157, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_45, (1536, ), (1, ))
    assert_size_stride(view_68, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_28, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(mean_3, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_3, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_30, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_169, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(squeeze_47, (768, ), (1, ))
    assert_size_stride(view_71, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_31, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_176, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_49, (768, ), (1, ))
    assert_size_stride(view_74, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_32, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_183, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_51, (768, ), (1, ))
    assert_size_stride(view_77, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_33, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_190, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_53, (1536, ), (1, ))
    assert_size_stride(view_80, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_34, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(mean_4, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_4, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_36, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_202, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(squeeze_55, (768, ), (1, ))
    assert_size_stride(view_83, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_37, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_209, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_57, (768, ), (1, ))
    assert_size_stride(view_86, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_38, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_216, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_59, (768, ), (1, ))
    assert_size_stride(view_89, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_39, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_223, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_61, (1536, ), (1, ))
    assert_size_stride(view_92, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_40, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(mean_5, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_5, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_42, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_235, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(squeeze_63, (768, ), (1, ))
    assert_size_stride(view_95, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_43, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_242, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_65, (768, ), (1, ))
    assert_size_stride(view_98, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_44, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_249, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_67, (768, ), (1, ))
    assert_size_stride(view_101, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_45, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_256, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_69, (1536, ), (1, ))
    assert_size_stride(view_104, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_46, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(mean_6, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_6, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_48, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_268, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(squeeze_71, (768, ), (1, ))
    assert_size_stride(view_107, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_49, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_275, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_73, (768, ), (1, ))
    assert_size_stride(view_110, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_50, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_282, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_75, (768, ), (1, ))
    assert_size_stride(view_113, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_51, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_289, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_77, (1536, ), (1, ))
    assert_size_stride(view_116, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_52, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(mean_7, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_7, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_54, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_301, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(squeeze_79, (768, ), (1, ))
    assert_size_stride(view_119, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_55, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_308, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_81, (768, ), (1, ))
    assert_size_stride(view_122, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_56, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_315, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_83, (768, ), (1, ))
    assert_size_stride(view_125, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_57, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(mul_322, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(squeeze_85, (1536, ), (1, ))
    assert_size_stride(view_128, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_58, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(mean_8, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_8, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_60, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_334, (4, 1536, 12, 12), (221184, 1, 18432, 1536))
    assert_size_stride(avg_pool2d_2, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(squeeze_87, (1536, ), (1, ))
    assert_size_stride(view_131, (1536, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_61, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(squeeze_89, (768, ), (1, ))
    assert_size_stride(view_134, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_62, (4, 768, 12, 12), (110592, 1, 9216, 768))
    assert_size_stride(constant_pad_nd_4, (4, 768, 13, 13), (129792, 1, 9984, 768))
    assert_size_stride(squeeze_91, (768, ), (1, ))
    assert_size_stride(view_137, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_63, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_351, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_93, (768, ), (1, ))
    assert_size_stride(view_140, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_64, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_358, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_95, (1536, ), (1, ))
    assert_size_stride(view_143, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_65, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(mean_9, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_9, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_67, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_370, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(squeeze_97, (768, ), (1, ))
    assert_size_stride(view_146, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_68, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_377, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_99, (768, ), (1, ))
    assert_size_stride(view_149, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_69, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_384, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_101, (768, ), (1, ))
    assert_size_stride(view_152, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_70, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_391, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_103, (1536, ), (1, ))
    assert_size_stride(view_155, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_71, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(mean_10, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_10, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_73, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_403, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(squeeze_105, (768, ), (1, ))
    assert_size_stride(view_158, (768, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_74, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_410, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_107, (768, ), (1, ))
    assert_size_stride(view_161, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_75, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_417, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_109, (768, ), (1, ))
    assert_size_stride(view_164, (768, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_76, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(mul_424, (4, 768, 6, 6), (27648, 1, 4608, 768))
    assert_size_stride(squeeze_111, (1536, ), (1, ))
    assert_size_stride(view_167, (1536, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_77, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(mean_11, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_11, (4, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_79, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(add_118, (4, 1536, 6, 6), (55296, 1, 9216, 1536))
    assert_size_stride(squeeze_113, (3072, ), (1, ))
    assert_size_stride(view_170, (3072, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_80, (4, 3072, 6, 6), (110592, 1, 18432, 3072))
    assert_size_stride(clone_12, (4, 3072), (3072, 1))
    assert_size_stride(permute_1, (1000, 3072), (3072, 1))
    assert_size_stride(unsqueeze_58, (1, 3072, 1), (3072, 1, 1))
    assert_size_stride(unsqueeze_66, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_74, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_82, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_90, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_98, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_106, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_114, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_122, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_130, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_138, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_146, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_154, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_162, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_170, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_186, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_194, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_210, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_218, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_234, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_266, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 768, 1), (768, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 32, 1), (32, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 16, 1), (16, 1, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty_strided((4, 512, 24, 24), (294912, 1, 12288, 512), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    buf221 = empty_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    buf319 = empty_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_mul_sigmoid_0(c_void_p(convolution_21.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf4.data_ptr()))
    buf5 = empty((4, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf5)
    del permute_1
    buf6 = empty((1000, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone_12, out=buf6)
    del clone_12
    buf7 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf8 = empty((4, 3072, 6, 6), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_gelu_gelu_backward_mul_sum_view_1(c_void_p(tangents_1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf5
    del convolution_80
    del tangents_1
    # Source Nodes: [gelu_51], Original ATen: [aten.convolution_backward, aten.div, aten.gelu, aten.gelu_backward, aten.mul]
    buf9 = aten.convolution_backward(buf8, add_118, view_170, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_118
    del view_170
    buf10 = buf9[0]
    buf11 = buf9[1]
    buf12 = buf9[2]
    del buf9
    buf13 = empty((3072, ), device='cpu', dtype=torch.float32)
    buf14 = empty((3072, ), device='cpu', dtype=torch.float32)
    buf15 = empty((3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf11, (3072, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf11  # reuse
    buf17 = empty((), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((4, 1536, 1, 1), (1536, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf18, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf18  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_2(c_void_p(buf16.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(unsqueeze_58.data_ptr()), c_void_p(squeeze_113.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()))
    del buf13
    del buf14
    del convolution_77
    del primals_181
    del primals_182
    del squeeze_113
    del unsqueeze_58
    # Source Nodes: [sigmoid_11], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf20 = aten.convolution_backward(buf19, relu_11, primals_230, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf19
    del primals_230
    buf21 = buf20[0]
    buf22 = buf20[1]
    buf23 = buf20[2]
    del buf20
    buf24 = buf21; del buf21  # reuse
    cpp_fused_convolution_backward_threshold_backward_3(c_void_p(buf24.data_ptr()), c_void_p(relu_11.data_ptr()))
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf25 = aten.convolution_backward(buf24, mean_11, primals_228, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf24
    del mean_11
    del primals_228
    buf26 = buf25[0]
    buf27 = buf25[1]
    buf28 = buf25[2]
    del buf25
    buf29 = empty_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_mul_sigmoid_4(c_void_p(buf10.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf29.data_ptr()))
    del convolution_79
    del primals_180
    # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf30 = aten.convolution_backward(buf29, mul_424, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf29
    del mul_424
    del view_167
    buf31 = buf30[0]
    buf32 = buf30[1]
    buf33 = buf30[2]
    del buf30
    buf34 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf35 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf36 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf37 = reinterpret_tensor(buf32, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf32  # reuse
    buf38 = buf31; del buf31  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_5(c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(unsqueeze_66.data_ptr()), c_void_p(squeeze_111.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del convolution_76
    del primals_177
    del primals_178
    del squeeze_111
    del unsqueeze_66
    # Source Nodes: [gelu_50], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf39 = aten.convolution_backward(buf38, mul_417, view_164, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf38
    del mul_417
    del view_164
    buf40 = buf39[0]
    buf41 = buf39[1]
    buf42 = buf39[2]
    del buf39
    buf43 = empty((768, ), device='cpu', dtype=torch.float32)
    buf44 = empty((768, ), device='cpu', dtype=torch.float32)
    buf45 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf46 = empty((768, 128, 3, 3), device='cpu', dtype=torch.float32)
    buf47 = buf40; del buf40  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_6(c_void_p(buf47.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(unsqueeze_74.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del convolution_75
    del primals_174
    del primals_175
    del squeeze_109
    del unsqueeze_74
    # Source Nodes: [gelu_49], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf48 = aten.convolution_backward(buf47, mul_410, view_161, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf47
    del mul_410
    del view_161
    buf49 = buf48[0]
    buf50 = buf48[1]
    buf51 = buf48[2]
    del buf48
    buf52 = buf44; del buf44  # reuse
    buf53 = buf43; del buf43  # reuse
    buf54 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf55 = reinterpret_tensor(buf41, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf41  # reuse
    buf56 = buf49; del buf49  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_7(c_void_p(buf56.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(unsqueeze_82.data_ptr()), c_void_p(squeeze_107.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del convolution_74
    del primals_171
    del primals_172
    del squeeze_107
    del unsqueeze_82
    # Source Nodes: [gelu_48], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf57 = aten.convolution_backward(buf56, mul_403, view_158, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf56
    del mul_403
    del view_158
    buf58 = buf57[0]
    buf59 = buf57[1]
    buf60 = buf57[2]
    del buf57
    buf61 = buf53; del buf53  # reuse
    buf62 = buf52; del buf52  # reuse
    buf63 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf64 = reinterpret_tensor(buf59, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf59  # reuse
    buf65 = buf10; del buf10  # reuse
    buf66 = empty((), device='cpu', dtype=torch.float32)
    buf67 = reinterpret_tensor(buf26, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf26  # reuse
    buf68 = reinterpret_tensor(buf67, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf67  # reuse
    cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_8(c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(unsqueeze_90.data_ptr()), c_void_p(squeeze_105.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf66.data_ptr()))
    del buf4
    del convolution_71
    del primals_168
    del primals_169
    del squeeze_105
    del unsqueeze_90
    # Source Nodes: [sigmoid_10], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf69 = aten.convolution_backward(buf68, relu_10, primals_226, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf68
    del primals_226
    buf70 = buf69[0]
    buf71 = buf69[1]
    buf72 = buf69[2]
    del buf69
    buf73 = buf70; del buf70  # reuse
    cpp_fused_convolution_backward_threshold_backward_9(c_void_p(buf73.data_ptr()), c_void_p(relu_10.data_ptr()))
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf74 = aten.convolution_backward(buf73, mean_10, primals_224, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf73
    del mean_10
    del primals_224
    buf75 = buf74[0]
    buf76 = buf74[1]
    buf77 = buf74[2]
    del buf74
    buf78 = buf58; del buf58  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_10(c_void_p(buf65.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf78.data_ptr()))
    del convolution_73
    del primals_167
    # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf79 = aten.convolution_backward(buf78, mul_391, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_391
    del view_155
    buf80 = buf79[0]
    buf81 = buf79[1]
    buf82 = buf79[2]
    del buf79
    buf83 = buf35; del buf35  # reuse
    buf84 = buf34; del buf34  # reuse
    buf85 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf81, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf81  # reuse
    buf87 = buf80; del buf80  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_11(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(unsqueeze_98.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    del convolution_70
    del primals_164
    del primals_165
    del squeeze_103
    del unsqueeze_98
    # Source Nodes: [gelu_46], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf88 = aten.convolution_backward(buf87, mul_384, view_152, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf87
    del mul_384
    del view_152
    buf89 = buf88[0]
    buf90 = buf88[1]
    buf91 = buf88[2]
    del buf88
    buf92 = buf62; del buf62  # reuse
    buf93 = buf61; del buf61  # reuse
    buf94 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf95 = reinterpret_tensor(buf50, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf50  # reuse
    buf96 = buf89; del buf89  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_12(c_void_p(buf96.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(unsqueeze_106.data_ptr()), c_void_p(squeeze_101.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del convolution_69
    del primals_161
    del primals_162
    del squeeze_101
    del unsqueeze_106
    # Source Nodes: [gelu_45], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf97 = aten.convolution_backward(buf96, mul_377, view_149, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf96
    del mul_377
    del view_149
    buf98 = buf97[0]
    buf99 = buf97[1]
    buf100 = buf97[2]
    del buf97
    buf101 = buf93; del buf93  # reuse
    buf102 = buf92; del buf92  # reuse
    buf103 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf104 = reinterpret_tensor(buf90, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf90  # reuse
    buf105 = buf98; del buf98  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_13(c_void_p(buf105.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(unsqueeze_114.data_ptr()), c_void_p(squeeze_99.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del convolution_68
    del primals_158
    del primals_159
    del squeeze_99
    del unsqueeze_114
    # Source Nodes: [gelu_44], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf106 = aten.convolution_backward(buf105, mul_370, view_146, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf105
    del mul_370
    del view_146
    buf107 = buf106[0]
    buf108 = buf106[1]
    buf109 = buf106[2]
    del buf106
    buf110 = buf102; del buf102  # reuse
    buf111 = buf101; del buf101  # reuse
    buf112 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf113 = reinterpret_tensor(buf108, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf108  # reuse
    buf114 = buf78; del buf78  # reuse
    buf115 = buf107; del buf107  # reuse
    buf116 = empty((), device='cpu', dtype=torch.float32)
    buf117 = reinterpret_tensor(buf75, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf75  # reuse
    buf118 = reinterpret_tensor(buf117, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf117  # reuse
    cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_14(c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(unsqueeze_122.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del buf114
    del convolution_61
    del convolution_65
    del primals_155
    del primals_156
    del squeeze_97
    del unsqueeze_122
    # Source Nodes: [sigmoid_9], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf119 = aten.convolution_backward(buf118, relu_9, primals_222, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf118
    del primals_222
    buf120 = buf119[0]
    buf121 = buf119[1]
    buf122 = buf119[2]
    del buf119
    buf123 = buf120; del buf120  # reuse
    cpp_fused_convolution_backward_threshold_backward_15(c_void_p(buf123.data_ptr()), c_void_p(relu_9.data_ptr()))
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf124 = aten.convolution_backward(buf123, mean_9, primals_220, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf123
    del mean_9
    del primals_220
    buf125 = buf124[0]
    buf126 = buf124[1]
    buf127 = buf124[2]
    del buf124
    buf128 = buf65; del buf65  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_16(c_void_p(buf115.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf128.data_ptr()))
    del convolution_67
    del primals_154
    # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf129 = aten.convolution_backward(buf128, mul_358, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf128
    del mul_358
    del view_143
    buf130 = buf129[0]
    buf131 = buf129[1]
    buf132 = buf129[2]
    del buf129
    buf133 = buf84; del buf84  # reuse
    buf134 = buf83; del buf83  # reuse
    buf135 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf136 = reinterpret_tensor(buf131, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf131  # reuse
    buf137 = buf130; del buf130  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_17(c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(unsqueeze_130.data_ptr()), c_void_p(squeeze_95.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del convolution_64
    del primals_151
    del primals_152
    del squeeze_95
    del unsqueeze_130
    # Source Nodes: [gelu_42], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf138 = aten.convolution_backward(buf137, mul_351, view_140, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf137
    del mul_351
    del view_140
    buf139 = buf138[0]
    buf140 = buf138[1]
    buf141 = buf138[2]
    del buf138
    buf142 = buf111; del buf111  # reuse
    buf143 = buf110; del buf110  # reuse
    buf144 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf145 = reinterpret_tensor(buf99, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf99  # reuse
    buf146 = buf139; del buf139  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_18(c_void_p(buf146.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(unsqueeze_138.data_ptr()), c_void_p(squeeze_93.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del convolution_63
    del primals_148
    del primals_149
    del squeeze_93
    del unsqueeze_138
    # Source Nodes: [gelu_41], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf147 = aten.convolution_backward(buf146, constant_pad_nd_4, view_137, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf146
    del constant_pad_nd_4
    del view_137
    buf148 = buf147[0]
    buf149 = buf147[1]
    buf150 = buf147[2]
    del buf147
    buf151 = buf143; del buf143  # reuse
    buf152 = buf142; del buf142  # reuse
    buf153 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf154 = reinterpret_tensor(buf140, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf140  # reuse
    buf155 = reinterpret_tensor(buf8, (4, 768, 12, 12), (110592, 1, 9216, 768), 0); del buf8  # reuse
    cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_19(c_void_p(buf149.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(unsqueeze_146.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del buf148
    del buf149
    del convolution_62
    del primals_145
    del primals_146
    del squeeze_91
    del unsqueeze_146
    # Source Nodes: [gelu_40], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf156 = aten.convolution_backward(buf155, mul_334, view_134, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf155
    del mul_334
    del view_134
    buf157 = buf156[0]
    buf158 = buf156[1]
    buf159 = buf156[2]
    del buf156
    buf160 = buf152; del buf152  # reuse
    buf161 = buf151; del buf151  # reuse
    buf162 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf163 = reinterpret_tensor(buf158, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf158  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_20(c_void_p(buf163.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(unsqueeze_154.data_ptr()), c_void_p(squeeze_89.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()))
    del primals_142
    del primals_143
    del squeeze_89
    del unsqueeze_154
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf164 = aten.convolution_backward(buf115, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del avg_pool2d_2
    del buf115
    del view_131
    buf165 = buf164[0]
    buf166 = buf164[1]
    buf167 = buf164[2]
    del buf164
    buf168 = buf134; del buf134  # reuse
    buf169 = buf133; del buf133  # reuse
    buf170 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf166, (1536, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf166  # reuse
    buf172 = buf157; del buf157  # reuse
    buf174 = reinterpret_tensor(buf125, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf125  # reuse
    buf175 = reinterpret_tensor(buf174, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf174  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_21(c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(unsqueeze_162.data_ptr()), c_void_p(squeeze_87.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()))
    del buf165
    del primals_139
    del primals_140
    del squeeze_87
    del unsqueeze_162
    # Source Nodes: [sigmoid_8], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf176 = aten.convolution_backward(buf175, relu_8, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf175
    del primals_218
    buf177 = buf176[0]
    buf180 = buf177; del buf177  # reuse
    cpp_fused_convolution_backward_threshold_backward_22(c_void_p(buf180.data_ptr()), c_void_p(relu_8.data_ptr()))
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf181 = aten.convolution_backward(buf180, mean_8, primals_216, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf180
    del mean_8
    del primals_216
    buf182 = buf181[0]
    buf185 = buf3; del buf3  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_23(c_void_p(buf172.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf185.data_ptr()))
    del primals_138
    # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf186 = aten.convolution_backward(buf185, mul_322, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_322
    del view_128
    buf187 = buf186[0]
    buf194 = buf187; del buf187  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_24(c_void_p(buf194.data_ptr()), c_void_p(convolution_57.data_ptr()))
    del convolution_57
    # Source Nodes: [gelu_38], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf195 = aten.convolution_backward(buf194, mul_315, view_125, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf194
    del mul_315
    del view_125
    buf196 = buf195[0]
    buf203 = buf196; del buf196  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_25(c_void_p(buf203.data_ptr()), c_void_p(convolution_56.data_ptr()))
    del convolution_56
    # Source Nodes: [gelu_37], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf204 = aten.convolution_backward(buf203, mul_308, view_122, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf203
    del mul_308
    del view_122
    buf205 = buf204[0]
    buf212 = buf205; del buf205  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_26(c_void_p(buf212.data_ptr()), c_void_p(convolution_55.data_ptr()))
    del convolution_55
    # Source Nodes: [gelu_36], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf213 = aten.convolution_backward(buf212, mul_301, view_119, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf212
    del mul_301
    del view_119
    buf214 = buf213[0]
    buf173 = empty((), device='cpu', dtype=torch.float32)
    buf222 = empty((), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_backward_mul_sigmoid_sum_27(c_void_p(buf172.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf222.data_ptr()))
    del convolution_58
    del convolution_60
    buf178 = buf176[1]
    buf179 = buf176[2]
    del buf176
    buf183 = buf181[1]
    buf184 = buf181[2]
    del buf181
    buf188 = buf186[1]
    buf189 = buf186[2]
    del buf186
    buf190 = buf169; del buf169  # reuse
    buf191 = buf168; del buf168  # reuse
    buf192 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf193 = reinterpret_tensor(buf188, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf188  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_28(c_void_p(buf193.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(unsqueeze_170.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del primals_135
    del primals_136
    del squeeze_85
    del unsqueeze_170
    buf197 = buf195[1]
    buf198 = buf195[2]
    del buf195
    buf199 = buf161; del buf161  # reuse
    buf200 = buf160; del buf160  # reuse
    buf201 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf202 = reinterpret_tensor(buf185, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf185  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_29(c_void_p(buf197.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(unsqueeze_178.data_ptr()), c_void_p(squeeze_83.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    del primals_132
    del primals_133
    del squeeze_83
    del unsqueeze_178
    buf206 = buf204[1]
    buf207 = buf204[2]
    del buf204
    buf208 = buf200; del buf200  # reuse
    buf209 = buf199; del buf199  # reuse
    buf210 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf211 = reinterpret_tensor(buf197, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf197  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_30(c_void_p(buf206.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(unsqueeze_186.data_ptr()), c_void_p(squeeze_81.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_129
    del primals_130
    del squeeze_81
    del unsqueeze_186
    buf215 = buf213[1]
    buf216 = buf213[2]
    del buf213
    buf217 = buf209; del buf209  # reuse
    buf218 = buf208; del buf208  # reuse
    buf219 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf220 = reinterpret_tensor(buf215, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf215  # reuse
    buf223 = reinterpret_tensor(buf182, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf182  # reuse
    buf224 = reinterpret_tensor(buf223, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf223  # reuse
    cpp_fused_add_convolution_backward_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_31(c_void_p(buf220.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(unsqueeze_194.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    del convolution_52
    del primals_126
    del primals_127
    del squeeze_79
    del unsqueeze_194
    # Source Nodes: [sigmoid_7], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf225 = aten.convolution_backward(buf224, relu_7, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf224
    del primals_214
    buf226 = buf225[0]
    buf227 = buf225[1]
    buf228 = buf225[2]
    del buf225
    buf229 = buf226; del buf226  # reuse
    cpp_fused_convolution_backward_threshold_backward_32(c_void_p(buf229.data_ptr()), c_void_p(relu_7.data_ptr()))
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf230 = aten.convolution_backward(buf229, mean_7, primals_212, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf229
    del mean_7
    del primals_212
    buf231 = buf230[0]
    buf232 = buf230[1]
    buf233 = buf230[2]
    del buf230
    buf234 = reinterpret_tensor(buf206, (4, 1536, 12, 12), (221184, 1, 18432, 1536), 0); del buf206  # reuse
    cpp_fused_add_convolution_backward_div_gelu_backward_mul_sigmoid_33(c_void_p(buf172.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf234.data_ptr()))
    del convolution_54
    del primals_125
    # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.gelu_backward, aten.mul, aten.sigmoid]
    buf235 = aten.convolution_backward(buf234, mul_289, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_289
    del view_116
    buf236 = buf235[0]
    buf237 = buf235[1]
    buf238 = buf235[2]
    del buf235
    buf239 = buf191; del buf191  # reuse
    buf240 = buf190; del buf190  # reuse
    buf241 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf242 = reinterpret_tensor(buf237, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf237  # reuse
    buf243 = buf236; del buf236  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_34(c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(unsqueeze_202.data_ptr()), c_void_p(squeeze_77.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del convolution_51
    del primals_122
    del primals_123
    del squeeze_77
    del unsqueeze_202
    # Source Nodes: [gelu_34], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf244 = aten.convolution_backward(buf243, mul_282, view_113, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf243
    del mul_282
    del view_113
    buf245 = buf244[0]
    buf246 = buf244[1]
    buf247 = buf244[2]
    del buf244
    buf248 = buf218; del buf218  # reuse
    buf249 = buf217; del buf217  # reuse
    buf250 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf251 = reinterpret_tensor(buf234, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf234  # reuse
    buf252 = buf245; del buf245  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_35(c_void_p(buf252.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_75.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del convolution_50
    del primals_119
    del primals_120
    del squeeze_75
    del unsqueeze_210
    # Source Nodes: [gelu_33], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf253 = aten.convolution_backward(buf252, mul_275, view_110, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf252
    del mul_275
    del view_110
    buf254 = buf253[0]
    buf255 = buf253[1]
    buf256 = buf253[2]
    del buf253
    buf257 = buf249; del buf249  # reuse
    buf258 = buf248; del buf248  # reuse
    buf259 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf260 = reinterpret_tensor(buf246, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf246  # reuse
    buf261 = buf254; del buf254  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_36(c_void_p(buf261.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(unsqueeze_218.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del buf255
    del convolution_49
    del primals_116
    del primals_117
    del squeeze_73
    del unsqueeze_218
    # Source Nodes: [gelu_32], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf262 = aten.convolution_backward(buf261, mul_268, view_107, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf261
    del mul_268
    del view_107
    buf263 = buf262[0]
    buf264 = buf262[1]
    buf265 = buf262[2]
    del buf262
    buf266 = buf258; del buf258  # reuse
    buf267 = buf257; del buf257  # reuse
    buf268 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf269 = reinterpret_tensor(buf264, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf264  # reuse
    buf270 = buf172; del buf172  # reuse
    buf272 = reinterpret_tensor(buf231, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf231  # reuse
    buf273 = reinterpret_tensor(buf272, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf272  # reuse
    cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_37(c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(unsqueeze_226.data_ptr()), c_void_p(squeeze_71.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    del buf2
    del buf214
    del buf221
    del primals_113
    del primals_114
    del squeeze_71
    del unsqueeze_226
    # Source Nodes: [sigmoid_6], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf274 = aten.convolution_backward(buf273, relu_6, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf273
    del primals_210
    buf275 = buf274[0]
    buf278 = buf275; del buf275  # reuse
    cpp_fused_convolution_backward_threshold_backward_38(c_void_p(buf278.data_ptr()), c_void_p(relu_6.data_ptr()))
    del relu_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf279 = aten.convolution_backward(buf278, mean_6, primals_208, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf278
    del mean_6
    del primals_208
    buf280 = buf279[0]
    buf283 = buf263; del buf263  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_39(c_void_p(buf270.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf283.data_ptr()))
    del primals_112
    # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf284 = aten.convolution_backward(buf283, mul_256, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_256
    del view_104
    buf285 = buf284[0]
    buf292 = buf285; del buf285  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_40(c_void_p(buf292.data_ptr()), c_void_p(convolution_45.data_ptr()))
    del convolution_45
    # Source Nodes: [gelu_30], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf293 = aten.convolution_backward(buf292, mul_249, view_101, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf292
    del mul_249
    del view_101
    buf294 = buf293[0]
    buf301 = buf294; del buf294  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_41(c_void_p(buf301.data_ptr()), c_void_p(convolution_44.data_ptr()))
    del convolution_44
    # Source Nodes: [gelu_29], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf302 = aten.convolution_backward(buf301, mul_242, view_98, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf301
    del mul_242
    del view_98
    buf303 = buf302[0]
    buf310 = buf303; del buf303  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_42(c_void_p(buf310.data_ptr()), c_void_p(convolution_43.data_ptr()))
    del convolution_43
    # Source Nodes: [gelu_28], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf311 = aten.convolution_backward(buf310, mul_235, view_95, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf310
    del mul_235
    del view_95
    buf312 = buf311[0]
    buf271 = empty((), device='cpu', dtype=torch.float32)
    buf320 = empty((), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_backward_mul_sigmoid_sum_43(c_void_p(buf270.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf320.data_ptr()))
    del convolution_46
    del convolution_48
    buf276 = buf274[1]
    buf277 = buf274[2]
    del buf274
    buf281 = buf279[1]
    buf282 = buf279[2]
    del buf279
    buf286 = buf284[1]
    buf287 = buf284[2]
    del buf284
    buf288 = buf240; del buf240  # reuse
    buf289 = buf239; del buf239  # reuse
    buf290 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf291 = reinterpret_tensor(buf286, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf286  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_44(c_void_p(buf291.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_69.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()))
    del primals_109
    del primals_110
    del squeeze_69
    del unsqueeze_234
    buf295 = buf293[1]
    buf296 = buf293[2]
    del buf293
    buf297 = buf267; del buf267  # reuse
    buf298 = buf266; del buf266  # reuse
    buf299 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf300 = reinterpret_tensor(buf283, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf283  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_45(c_void_p(buf295.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(unsqueeze_242.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del primals_106
    del primals_107
    del squeeze_67
    del unsqueeze_242
    buf304 = buf302[1]
    buf305 = buf302[2]
    del buf302
    buf306 = buf298; del buf298  # reuse
    buf307 = buf297; del buf297  # reuse
    buf308 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf309 = reinterpret_tensor(buf295, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf295  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_46(c_void_p(buf304.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_65.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()))
    del primals_103
    del primals_104
    del squeeze_65
    del unsqueeze_250
    buf313 = buf311[1]
    buf314 = buf311[2]
    del buf311
    buf315 = buf307; del buf307  # reuse
    buf316 = buf306; del buf306  # reuse
    buf317 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf318 = reinterpret_tensor(buf313, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf313  # reuse
    buf321 = reinterpret_tensor(buf280, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf280  # reuse
    buf322 = reinterpret_tensor(buf321, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf321  # reuse
    cpp_fused_add_convolution_backward_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_47(c_void_p(buf318.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_63.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    del convolution_40
    del primals_100
    del primals_101
    del squeeze_63
    del unsqueeze_258
    # Source Nodes: [sigmoid_5], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf323 = aten.convolution_backward(buf322, relu_5, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf322
    del primals_206
    buf324 = buf323[0]
    buf325 = buf323[1]
    buf326 = buf323[2]
    del buf323
    buf327 = buf324; del buf324  # reuse
    cpp_fused_convolution_backward_threshold_backward_48(c_void_p(buf327.data_ptr()), c_void_p(relu_5.data_ptr()))
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf328 = aten.convolution_backward(buf327, mean_5, primals_204, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf327
    del mean_5
    del primals_204
    buf329 = buf328[0]
    buf330 = buf328[1]
    buf331 = buf328[2]
    del buf328
    buf332 = reinterpret_tensor(buf304, (4, 1536, 12, 12), (221184, 1, 18432, 1536), 0); del buf304  # reuse
    cpp_fused_add_convolution_backward_div_gelu_backward_mul_sigmoid_49(c_void_p(buf270.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf332.data_ptr()))
    del convolution_42
    del primals_99
    # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.gelu_backward, aten.mul, aten.sigmoid]
    buf333 = aten.convolution_backward(buf332, mul_223, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_223
    del view_92
    buf334 = buf333[0]
    buf335 = buf333[1]
    buf336 = buf333[2]
    del buf333
    buf337 = buf289; del buf289  # reuse
    buf338 = buf288; del buf288  # reuse
    buf339 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf340 = reinterpret_tensor(buf335, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf335  # reuse
    buf341 = buf334; del buf334  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_50(c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(unsqueeze_266.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    del convolution_39
    del primals_96
    del primals_97
    del squeeze_61
    del unsqueeze_266
    # Source Nodes: [gelu_26], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf342 = aten.convolution_backward(buf341, mul_216, view_89, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf341
    del mul_216
    del view_89
    buf343 = buf342[0]
    buf344 = buf342[1]
    buf345 = buf342[2]
    del buf342
    buf346 = buf316; del buf316  # reuse
    buf347 = buf315; del buf315  # reuse
    buf348 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf349 = reinterpret_tensor(buf332, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf332  # reuse
    buf350 = buf343; del buf343  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_51(c_void_p(buf350.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_59.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    del convolution_38
    del primals_93
    del primals_94
    del squeeze_59
    del unsqueeze_274
    # Source Nodes: [gelu_25], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf351 = aten.convolution_backward(buf350, mul_209, view_86, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf350
    del mul_209
    del view_86
    buf352 = buf351[0]
    buf353 = buf351[1]
    buf354 = buf351[2]
    del buf351
    buf355 = buf347; del buf347  # reuse
    buf356 = buf346; del buf346  # reuse
    buf357 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf358 = reinterpret_tensor(buf344, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf344  # reuse
    buf359 = buf352; del buf352  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_52(c_void_p(buf359.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_57.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    del buf353
    del convolution_37
    del primals_90
    del primals_91
    del squeeze_57
    del unsqueeze_282
    # Source Nodes: [gelu_24], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf360 = aten.convolution_backward(buf359, mul_202, view_83, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf359
    del mul_202
    del view_83
    buf361 = buf360[0]
    buf362 = buf360[1]
    buf363 = buf360[2]
    del buf360
    buf364 = buf356; del buf356  # reuse
    buf365 = buf355; del buf355  # reuse
    buf366 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf367 = reinterpret_tensor(buf362, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf362  # reuse
    buf368 = buf1; del buf1  # reuse
    buf369 = empty((), device='cpu', dtype=torch.float32)
    buf370 = reinterpret_tensor(buf329, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf329  # reuse
    buf371 = reinterpret_tensor(buf370, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf370  # reuse
    cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_53(c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(unsqueeze_290.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf369.data_ptr()))
    del buf270
    del buf312
    del buf319
    del convolution_34
    del primals_87
    del primals_88
    del squeeze_55
    del unsqueeze_290
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf372 = aten.convolution_backward(buf371, relu_4, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf371
    del primals_202
    buf373 = buf372[0]
    buf374 = buf372[1]
    buf375 = buf372[2]
    del buf372
    buf376 = buf373; del buf373  # reuse
    cpp_fused_convolution_backward_threshold_backward_54(c_void_p(buf376.data_ptr()), c_void_p(relu_4.data_ptr()))
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf377 = aten.convolution_backward(buf376, mean_4, primals_200, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf376
    del mean_4
    del primals_200
    buf378 = buf377[0]
    buf379 = buf377[1]
    buf380 = buf377[2]
    del buf377
    buf381 = buf361; del buf361  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_55(c_void_p(buf368.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf381.data_ptr()))
    del convolution_36
    del primals_86
    # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf382 = aten.convolution_backward(buf381, mul_190, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_190
    del view_80
    buf383 = buf382[0]
    buf384 = buf382[1]
    buf385 = buf382[2]
    del buf382
    buf386 = buf338; del buf338  # reuse
    buf387 = buf337; del buf337  # reuse
    buf388 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf389 = reinterpret_tensor(buf384, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf384  # reuse
    buf390 = buf383; del buf383  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_56(c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_53.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()))
    del convolution_33
    del primals_83
    del primals_84
    del squeeze_53
    del unsqueeze_298
    # Source Nodes: [gelu_22], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf391 = aten.convolution_backward(buf390, mul_183, view_77, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf390
    del mul_183
    del view_77
    buf392 = buf391[0]
    buf393 = buf391[1]
    buf394 = buf391[2]
    del buf391
    buf395 = buf365; del buf365  # reuse
    buf396 = buf364; del buf364  # reuse
    buf397 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf398 = reinterpret_tensor(buf381, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf381  # reuse
    buf399 = buf392; del buf392  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_57(c_void_p(buf399.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_51.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()))
    del convolution_32
    del primals_80
    del primals_81
    del squeeze_51
    del unsqueeze_306
    # Source Nodes: [gelu_21], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf400 = aten.convolution_backward(buf399, mul_176, view_74, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf399
    del mul_176
    del view_74
    buf401 = buf400[0]
    buf402 = buf400[1]
    buf403 = buf400[2]
    del buf400
    buf404 = buf396; del buf396  # reuse
    buf405 = buf395; del buf395  # reuse
    buf406 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf407 = reinterpret_tensor(buf393, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf393  # reuse
    buf408 = buf401; del buf401  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_58(c_void_p(buf408.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(unsqueeze_314.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    del convolution_31
    del primals_77
    del primals_78
    del squeeze_49
    del unsqueeze_314
    # Source Nodes: [gelu_20], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf409 = aten.convolution_backward(buf408, mul_169, view_71, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf408
    del mul_169
    del view_71
    buf410 = buf409[0]
    buf411 = buf409[1]
    buf412 = buf409[2]
    del buf409
    buf413 = buf405; del buf405  # reuse
    buf414 = buf404; del buf404  # reuse
    buf415 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf416 = reinterpret_tensor(buf411, (768, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf411  # reuse
    buf417 = reinterpret_tensor(buf402, (4, 1536, 12, 12), (221184, 1, 18432, 1536), 0); del buf402  # reuse
    buf418 = buf368; del buf368  # reuse
    buf419 = empty((), device='cpu', dtype=torch.float32)
    buf420 = reinterpret_tensor(buf378, (4, 1536, 1, 1), (1536, 1, 6144, 6144), 0); del buf378  # reuse
    buf421 = reinterpret_tensor(buf420, (4, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf420  # reuse
    cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_59(c_void_p(buf416.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_47.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()))
    del buf410
    del convolution_24
    del convolution_28
    del primals_74
    del primals_75
    del squeeze_47
    del unsqueeze_322
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf422 = aten.convolution_backward(buf421, relu_3, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf421
    del primals_198
    buf423 = buf422[0]
    buf424 = buf422[1]
    buf425 = buf422[2]
    del buf422
    buf426 = buf423; del buf423  # reuse
    cpp_fused_convolution_backward_threshold_backward_60(c_void_p(buf426.data_ptr()), c_void_p(relu_3.data_ptr()))
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf427 = aten.convolution_backward(buf426, mean_3, primals_196, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf426
    del mean_3
    del primals_196
    buf428 = buf427[0]
    buf429 = buf427[1]
    buf430 = buf427[2]
    del buf427
    buf431 = buf417; del buf417  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_61(c_void_p(buf418.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf431.data_ptr()))
    del buf428
    del convolution_30
    del primals_73
    # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf432 = aten.convolution_backward(buf431, mul_157, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_157
    del view_68
    buf433 = buf432[0]
    buf434 = buf432[1]
    buf435 = buf432[2]
    del buf432
    buf436 = buf387; del buf387  # reuse
    buf437 = buf386; del buf386  # reuse
    buf438 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf439 = reinterpret_tensor(buf434, (1536, 768, 1, 1), (768, 1, 1, 1), 0); del buf434  # reuse
    buf440 = buf433; del buf433  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_62(c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_45.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    del convolution_27
    del primals_70
    del primals_71
    del squeeze_45
    del unsqueeze_330
    # Source Nodes: [gelu_18], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf441 = aten.convolution_backward(buf440, mul_150, view_65, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf440
    del mul_150
    del view_65
    buf442 = buf441[0]
    buf443 = buf441[1]
    buf444 = buf441[2]
    del buf441
    buf445 = buf414; del buf414  # reuse
    buf446 = buf413; del buf413  # reuse
    buf447 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf448 = reinterpret_tensor(buf431, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf431  # reuse
    buf449 = buf442; del buf442  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_63(c_void_p(buf449.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(unsqueeze_338.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    del convolution_26
    del primals_67
    del primals_68
    del squeeze_43
    del unsqueeze_338
    # Source Nodes: [gelu_17], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf450 = aten.convolution_backward(buf449, constant_pad_nd_3, view_62, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf449
    del constant_pad_nd_3
    del view_62
    buf451 = buf450[0]
    buf452 = buf450[1]
    buf453 = buf450[2]
    del buf450
    buf454 = buf446; del buf446  # reuse
    buf455 = buf445; del buf445  # reuse
    buf456 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf457 = reinterpret_tensor(buf443, (768, 128, 3, 3), (1152, 9, 3, 1), 0); del buf443  # reuse
    buf458 = empty_strided((4, 768, 24, 24), (442368, 1, 18432, 768), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_64(c_void_p(buf452.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_41.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    del buf451
    del buf452
    del convolution_25
    del primals_64
    del primals_65
    del squeeze_41
    del unsqueeze_346
    # Source Nodes: [gelu_16], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf459 = aten.convolution_backward(buf458, mul_133, view_59, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf458
    del mul_133
    del view_59
    buf460 = buf459[0]
    buf461 = buf459[1]
    buf462 = buf459[2]
    del buf459
    buf463 = buf455; del buf455  # reuse
    buf464 = buf454; del buf454  # reuse
    buf465 = empty((768, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf466 = reinterpret_tensor(buf461, (768, 512, 1, 1), (512, 1, 1, 1), 0); del buf461  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_65(c_void_p(buf466.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_39.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()))
    del buf463
    del buf464
    del primals_61
    del primals_62
    del squeeze_39
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf467 = aten.convolution_backward(buf418, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del avg_pool2d_1
    del buf418
    del view_56
    buf468 = buf467[0]
    buf469 = buf467[1]
    buf470 = buf467[2]
    del buf467
    buf471 = buf437; del buf437  # reuse
    buf472 = buf436; del buf436  # reuse
    buf473 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf474 = reinterpret_tensor(buf469, (1536, 512, 1, 1), (512, 1, 1, 1), 0); del buf469  # reuse
    buf475 = buf0; del buf0  # reuse
    buf476 = empty((), device='cpu', dtype=torch.float32)
    buf477 = empty_strided((4, 512, 1, 1), (512, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf478 = reinterpret_tensor(buf477, (4, 512, 1, 1), (512, 1, 1, 1), 0); del buf477  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_66(c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(unsqueeze_362.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf476.data_ptr()))
    del buf471
    del buf472
    del convolution_21
    del primals_58
    del primals_59
    del squeeze_37
    del unsqueeze_362
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf479 = aten.convolution_backward(buf478, relu_2, primals_194, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf478
    del primals_194
    buf480 = buf479[0]
    buf481 = buf479[1]
    buf482 = buf479[2]
    del buf479
    buf483 = buf480; del buf480  # reuse
    cpp_fused_convolution_backward_threshold_backward_67(c_void_p(buf483.data_ptr()), c_void_p(relu_2.data_ptr()))
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf484 = aten.convolution_backward(buf483, mean_2, primals_192, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf483
    del mean_2
    del primals_192
    buf485 = buf484[0]
    buf486 = buf484[1]
    buf487 = buf484[2]
    del buf484
    buf488 = buf460; del buf460  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_68(c_void_p(buf475.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf488.data_ptr()))
    del convolution_23
    del primals_57
    # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf489 = aten.convolution_backward(buf488, mul_121, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_121
    del view_53
    buf490 = buf489[0]
    buf491 = buf489[1]
    buf492 = buf489[2]
    del buf489
    buf493 = empty((512, ), device='cpu', dtype=torch.float32)
    buf494 = empty((512, ), device='cpu', dtype=torch.float32)
    buf495 = empty((512, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf496 = reinterpret_tensor(buf491, (512, 256, 1, 1), (256, 1, 1, 1), 0); del buf491  # reuse
    buf497 = buf490; del buf490  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_69(c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_35.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    del convolution_20
    del primals_54
    del primals_55
    del squeeze_35
    del unsqueeze_370
    # Source Nodes: [gelu_14], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf498 = aten.convolution_backward(buf497, mul_114, view_50, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf497
    del mul_114
    del view_50
    buf499 = buf498[0]
    buf500 = buf498[1]
    buf501 = buf498[2]
    del buf498
    buf502 = empty((256, ), device='cpu', dtype=torch.float32)
    buf503 = empty((256, ), device='cpu', dtype=torch.float32)
    buf504 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf505 = reinterpret_tensor(buf468, (256, 128, 3, 3), (1152, 9, 3, 1), 0); del buf468  # reuse
    buf506 = buf499; del buf499  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_70(c_void_p(buf506.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_33.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()))
    del convolution_19
    del primals_51
    del primals_52
    del squeeze_33
    del unsqueeze_378
    # Source Nodes: [gelu_13], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf507 = aten.convolution_backward(buf506, mul_107, view_47, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf506
    del mul_107
    del view_47
    buf508 = buf507[0]
    buf509 = buf507[1]
    buf510 = buf507[2]
    del buf507
    buf511 = buf503; del buf503  # reuse
    buf512 = buf502; del buf502  # reuse
    buf513 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf514 = reinterpret_tensor(buf500, (256, 128, 3, 3), (1152, 9, 3, 1), 0); del buf500  # reuse
    buf515 = buf508; del buf508  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_71(c_void_p(buf515.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(unsqueeze_386.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf514.data_ptr()))
    del convolution_18
    del primals_48
    del primals_49
    del squeeze_31
    del unsqueeze_386
    # Source Nodes: [gelu_12], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf516 = aten.convolution_backward(buf515, mul_100, view_44, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf515
    del mul_100
    del view_44
    buf517 = buf516[0]
    buf518 = buf516[1]
    buf519 = buf516[2]
    del buf516
    buf520 = buf512; del buf512  # reuse
    buf521 = buf511; del buf511  # reuse
    buf522 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf523 = reinterpret_tensor(buf518, (256, 512, 1, 1), (512, 1, 1, 1), 0); del buf518  # reuse
    buf524 = buf488; del buf488  # reuse
    buf525 = buf475; del buf475  # reuse
    buf526 = empty((), device='cpu', dtype=torch.float32)
    buf527 = reinterpret_tensor(buf485, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf485  # reuse
    buf528 = reinterpret_tensor(buf527, (4, 512, 1, 1), (512, 1, 1, 1), 0); del buf527  # reuse
    cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_72(c_void_p(buf523.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_29.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf526.data_ptr()))
    del buf517
    del convolution_11
    del convolution_15
    del primals_45
    del primals_46
    del squeeze_29
    del unsqueeze_394
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf529 = aten.convolution_backward(buf528, relu_1, primals_190, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf528
    del primals_190
    buf530 = buf529[0]
    buf531 = buf529[1]
    buf532 = buf529[2]
    del buf529
    buf533 = buf530; del buf530  # reuse
    cpp_fused_convolution_backward_threshold_backward_73(c_void_p(buf533.data_ptr()), c_void_p(relu_1.data_ptr()))
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf534 = aten.convolution_backward(buf533, mean_1, primals_188, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_1
    del primals_188
    buf535 = buf534[0]
    buf536 = buf534[1]
    buf537 = buf534[2]
    del buf534
    buf538 = buf524; del buf524  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_74(c_void_p(buf525.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf538.data_ptr()))
    del buf535
    del convolution_17
    del primals_44
    # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf539 = aten.convolution_backward(buf538, mul_88, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf538
    del mul_88
    del view_41
    buf540 = buf539[0]
    buf541 = buf539[1]
    buf542 = buf539[2]
    del buf539
    buf543 = buf494; del buf494  # reuse
    buf544 = buf493; del buf493  # reuse
    buf545 = empty((512, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf546 = reinterpret_tensor(buf541, (512, 256, 1, 1), (256, 1, 1, 1), 0); del buf541  # reuse
    buf547 = buf540; del buf540  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_75(c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_27.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()))
    del convolution_14
    del primals_41
    del primals_42
    del squeeze_27
    del unsqueeze_402
    # Source Nodes: [gelu_10], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf548 = aten.convolution_backward(buf547, mul_81, view_38, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf547
    del mul_81
    del view_38
    buf549 = buf548[0]
    buf550 = buf548[1]
    buf551 = buf548[2]
    del buf548
    buf552 = buf521; del buf521  # reuse
    buf553 = buf520; del buf520  # reuse
    buf554 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf555 = reinterpret_tensor(buf509, (256, 128, 3, 3), (1152, 9, 3, 1), 0); del buf509  # reuse
    buf556 = buf549; del buf549  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_76(c_void_p(buf556.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(unsqueeze_410.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()))
    del convolution_13
    del primals_38
    del primals_39
    del squeeze_25
    del unsqueeze_410
    # Source Nodes: [gelu_9], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf557 = aten.convolution_backward(buf556, constant_pad_nd_2, view_35, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf556
    del constant_pad_nd_2
    del view_35
    buf558 = buf557[0]
    buf559 = buf557[1]
    buf560 = buf557[2]
    del buf557
    buf561 = buf553; del buf553  # reuse
    buf562 = buf552; del buf552  # reuse
    buf563 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf564 = reinterpret_tensor(buf550, (256, 128, 3, 3), (1152, 9, 3, 1), 0); del buf550  # reuse
    buf565 = empty_strided((4, 256, 48, 48), (589824, 1, 12288, 256), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_77(c_void_p(buf559.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_23.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()))
    del buf558
    del buf559
    del convolution_12
    del primals_35
    del primals_36
    del squeeze_23
    del unsqueeze_418
    # Source Nodes: [gelu_8], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf566 = aten.convolution_backward(buf565, mul_64, view_32, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_64
    del view_32
    buf567 = buf566[0]
    buf568 = buf566[1]
    buf569 = buf566[2]
    del buf566
    buf570 = buf562; del buf562  # reuse
    buf571 = buf561; del buf561  # reuse
    buf572 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf573 = reinterpret_tensor(buf568, (256, 256, 1, 1), (256, 1, 1, 1), 0); del buf568  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_78(c_void_p(buf573.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_21.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()))
    del primals_32
    del primals_33
    del squeeze_21
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf574 = aten.convolution_backward(buf525, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del avg_pool2d
    del buf525
    del view_29
    buf575 = buf574[0]
    buf576 = buf574[1]
    buf577 = buf574[2]
    del buf574
    buf578 = buf544; del buf544  # reuse
    buf579 = buf543; del buf543  # reuse
    buf580 = empty((512, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf581 = reinterpret_tensor(buf576, (512, 256, 1, 1), (256, 1, 1, 1), 0); del buf576  # reuse
    buf582 = buf565; del buf565  # reuse
    buf583 = buf567; del buf567  # reuse
    buf584 = empty((), device='cpu', dtype=torch.float32)
    buf585 = reinterpret_tensor(buf533, (4, 256, 1, 1), (256, 1, 1024, 1024), 0); del buf533  # reuse
    buf586 = reinterpret_tensor(buf585, (4, 256, 1, 1), (256, 1, 1, 1), 0); del buf585  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_79(c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(unsqueeze_434.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf584.data_ptr()))
    del buf575
    del buf578
    del buf579
    del convolution_4
    del convolution_8
    del primals_29
    del primals_30
    del squeeze_19
    del unsqueeze_434
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf587 = aten.convolution_backward(buf586, relu, primals_186, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf586
    del primals_186
    buf588 = buf587[0]
    buf589 = buf587[1]
    buf590 = buf587[2]
    del buf587
    buf591 = buf588; del buf588  # reuse
    cpp_fused_convolution_backward_threshold_backward_80(c_void_p(buf591.data_ptr()), c_void_p(relu.data_ptr()))
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf592 = aten.convolution_backward(buf591, mean, primals_184, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf591
    del mean
    del primals_184
    buf593 = buf592[0]
    buf594 = buf592[1]
    buf595 = buf592[2]
    del buf592
    buf596 = buf582; del buf582  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_81(c_void_p(buf583.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf596.data_ptr()))
    del buf593
    del convolution_10
    del primals_28
    # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf597 = aten.convolution_backward(buf596, mul_52, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf596
    del mul_52
    del view_26
    buf598 = buf597[0]
    buf599 = buf597[1]
    buf600 = buf597[2]
    del buf597
    buf601 = buf571; del buf571  # reuse
    buf602 = buf570; del buf570  # reuse
    buf603 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf604 = reinterpret_tensor(buf599, (256, 128, 1, 1), (128, 1, 1, 1), 0); del buf599  # reuse
    buf605 = buf598; del buf598  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_82(c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_17.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()))
    del convolution_7
    del primals_25
    del primals_26
    del squeeze_17
    del unsqueeze_442
    # Source Nodes: [gelu_6], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf606 = aten.convolution_backward(buf605, mul_45, view_23, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf605
    del mul_45
    del view_23
    buf607 = buf606[0]
    buf608 = buf606[1]
    buf609 = buf606[2]
    del buf606
    buf610 = empty((128, ), device='cpu', dtype=torch.float32)
    buf611 = empty((128, ), device='cpu', dtype=torch.float32)
    buf612 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf613 = empty((128, 128, 3, 3), device='cpu', dtype=torch.float32)
    buf614 = buf607; del buf607  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_83(c_void_p(buf614.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_15.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()))
    del convolution_6
    del primals_22
    del primals_23
    del squeeze_15
    del unsqueeze_450
    # Source Nodes: [gelu_5], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf615 = aten.convolution_backward(buf614, mul_38, view_20, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf614
    del mul_38
    del view_20
    buf616 = buf615[0]
    buf617 = buf615[1]
    buf618 = buf615[2]
    del buf615
    buf619 = buf611; del buf611  # reuse
    buf620 = buf610; del buf610  # reuse
    buf621 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf622 = reinterpret_tensor(buf608, (128, 128, 3, 3), (1152, 9, 3, 1), 0); del buf608  # reuse
    buf623 = buf616; del buf616  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_84(c_void_p(buf623.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(unsqueeze_458.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()))
    del buf617
    del convolution_5
    del primals_19
    del primals_20
    del squeeze_13
    del unsqueeze_458
    # Source Nodes: [gelu_4], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf624 = aten.convolution_backward(buf623, mul_28, view_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf623
    del view_17
    buf625 = buf624[0]
    buf626 = buf624[1]
    buf627 = buf624[2]
    del buf624
    buf628 = buf620; del buf620  # reuse
    buf629 = buf619; del buf619  # reuse
    buf630 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf631 = reinterpret_tensor(buf626, (128, 128, 1, 1), (128, 1, 1, 1), 0); del buf626  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_85(c_void_p(buf631.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_11.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()))
    del primals_16
    del primals_17
    del squeeze_11
    del unsqueeze_466
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf632 = aten.convolution_backward(buf583, mul_28, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_28
    del view_14
    buf633 = buf632[0]
    buf634 = buf632[1]
    buf635 = buf632[2]
    del buf632
    buf636 = buf602; del buf602  # reuse
    buf637 = buf601; del buf601  # reuse
    buf638 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf639 = reinterpret_tensor(buf634, (256, 128, 1, 1), (128, 1, 1, 1), 0); del buf634  # reuse
    buf640 = buf625; del buf625  # reuse
    cpp_fused_add_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_86(c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_9.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()))
    del buf633
    del buf636
    del buf637
    del convolution_3
    del primals_13
    del primals_14
    del squeeze_9
    del unsqueeze_474
    # Source Nodes: [gelu_3], Original ATen: [aten.add, aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf641 = aten.convolution_backward(buf640, constant_pad_nd_1, view_11, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf640
    del constant_pad_nd_1
    del view_11
    buf642 = buf641[0]
    buf643 = buf641[1]
    buf644 = buf641[2]
    del buf641
    buf645 = buf629; del buf629  # reuse
    buf646 = buf628; del buf628  # reuse
    buf647 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf648 = empty((128, 64, 3, 3), device='cpu', dtype=torch.float32)
    buf649 = reinterpret_tensor(buf583, (4, 64, 96, 96), (589824, 1, 6144, 64), 0); del buf583  # reuse
    cpp_fused_constant_pad_nd_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_87(c_void_p(buf643.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(unsqueeze_482.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()))
    del buf642
    del buf643
    del buf645
    del buf646
    del convolution_2
    del primals_10
    del primals_11
    del squeeze_7
    del unsqueeze_482
    # Source Nodes: [gelu_2], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf650 = aten.convolution_backward(buf649, mul_13, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf649
    del mul_13
    del view_8
    buf651 = buf650[0]
    buf652 = buf650[1]
    buf653 = buf650[2]
    del buf650
    buf654 = empty((64, ), device='cpu', dtype=torch.float32)
    buf655 = empty((64, ), device='cpu', dtype=torch.float32)
    buf656 = empty((64, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf657 = empty((64, 32, 3, 3), device='cpu', dtype=torch.float32)
    buf658 = buf651; del buf651  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_88(c_void_p(buf658.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_5.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()))
    del buf652
    del buf654
    del buf655
    del convolution_1
    del primals_7
    del primals_8
    del squeeze_5
    del unsqueeze_490
    # Source Nodes: [gelu_1], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf659 = aten.convolution_backward(buf658, mul_6, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf658
    del mul_6
    del view_5
    buf660 = buf659[0]
    buf661 = buf659[1]
    buf662 = buf659[2]
    del buf659
    buf663 = empty((32, ), device='cpu', dtype=torch.float32)
    buf664 = empty((32, ), device='cpu', dtype=torch.float32)
    buf665 = empty((32, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf666 = empty((32, 16, 3, 3), device='cpu', dtype=torch.float32)
    buf667 = buf660; del buf660  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_mul_native_batch_norm_backward_view_89(c_void_p(buf667.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_3.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()))
    del buf661
    del buf663
    del buf664
    del convolution
    del primals_4
    del primals_5
    del squeeze_3
    del unsqueeze_498
    # Source Nodes: [gelu], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward, aten.mul]
    buf668 = aten.convolution_backward(buf667, constant_pad_nd, view_2, [16], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf667
    del constant_pad_nd
    del view_2
    buf669 = buf668[1]
    buf670 = buf668[2]
    del buf668
    buf671 = empty((16, ), device='cpu', dtype=torch.float32)
    buf672 = empty((16, ), device='cpu', dtype=torch.float32)
    buf673 = empty((16, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf674 = empty((16, 3, 3, 3), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_batch_norm_backward_view_90(c_void_p(buf669.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(unsqueeze_506.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf674.data_ptr()))
    del buf669
    del buf671
    del buf672
    del primals_1
    del primals_2
    del squeeze_1
    del unsqueeze_506
    return (buf674, buf673, buf670, buf666, buf665, buf662, buf657, buf656, buf653, buf648, buf647, buf644, buf639, buf638, buf635, buf631, buf630, buf627, buf622, buf621, buf618, buf613, buf612, buf609, buf604, buf603, buf600, buf584, buf581, buf580, buf577, buf573, buf572, buf569, buf564, buf563, buf560, buf555, buf554, buf551, buf546, buf545, buf542, buf526, buf523, buf522, buf519, buf514, buf513, buf510, buf505, buf504, buf501, buf496, buf495, buf492, buf476, buf474, buf473, buf470, buf466, buf465, buf462, buf457, buf456, buf453, buf448, buf447, buf444, buf439, buf438, buf435, buf419, buf416, buf415, buf412, buf407, buf406, buf403, buf398, buf397, buf394, buf389, buf388, buf385, buf369, buf367, buf366, buf363, buf358, buf357, buf354, buf349, buf348, buf345, buf340, buf339, buf336, buf320, buf318, buf317, buf314, buf309, buf308, buf305, buf300, buf299, buf296, buf291, buf290, buf287, buf271, buf269, buf268, buf265, buf260, buf259, buf256, buf251, buf250, buf247, buf242, buf241, buf238, buf222, buf220, buf219, buf216, buf211, buf210, buf207, buf202, buf201, buf198, buf193, buf192, buf189, buf173, buf171, buf170, buf167, buf163, buf162, buf159, buf154, buf153, buf150, buf145, buf144, buf141, buf136, buf135, buf132, buf116, buf113, buf112, buf109, buf104, buf103, buf100, buf95, buf94, buf91, buf86, buf85, buf82, buf66, buf64, buf63, buf60, buf55, buf54, buf51, buf46, buf45, buf42, buf37, buf36, buf33, buf17, buf16, buf15, buf12, buf594, buf595, buf589, buf590, buf536, buf537, buf531, buf532, buf486, buf487, buf481, buf482, buf429, buf430, buf424, buf425, buf379, buf380, buf374, buf375, buf330, buf331, buf325, buf326, buf281, buf282, buf276, buf277, buf232, buf233, buf227, buf228, buf183, buf184, buf178, buf179, buf126, buf127, buf121, buf122, buf76, buf77, buf71, buf72, buf27, buf28, buf22, buf23, reinterpret_tensor(buf6, (1000, 3072), (3072, 1), 0), buf7, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((3072, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    constant_pad_nd = rand_strided((4, 3, 193, 193), (111747, 1, 579, 3), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 16, 96, 96), (147456, 1, 1536, 16), device='cpu', dtype=torch.float32)
    mul_6 = rand_strided((4, 16, 96, 96), (147456, 1, 1536, 16), device='cpu', dtype=torch.float32)
    squeeze_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    mul_13 = rand_strided((4, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    squeeze_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    view_8 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 64, 96, 96), (589824, 1, 6144, 64), device='cpu', dtype=torch.float32)
    constant_pad_nd_1 = rand_strided((4, 64, 97, 97), (602176, 1, 6208, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    mul_28 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    squeeze_9 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_14 = rand_strided((256, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 256, 48, 48), (589824, 1, 12288, 256), device='cpu', dtype=torch.float32)
    squeeze_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((128, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    mul_38 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    mul_45 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    squeeze_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    mul_52 = rand_strided((4, 128, 48, 48), (294912, 1, 6144, 128), device='cpu', dtype=torch.float32)
    squeeze_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_26 = rand_strided((256, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 256, 48, 48), (589824, 1, 12288, 256), device='cpu', dtype=torch.float32)
    mean = rand_strided((4, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    mul_64 = rand_strided((4, 256, 48, 48), (589824, 1, 12288, 256), device='cpu', dtype=torch.float32)
    avg_pool2d = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 512, 24, 24), (294912, 1, 12288, 512), device='cpu', dtype=torch.float32)
    squeeze_21 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((256, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 256, 48, 48), (589824, 1, 12288, 256), device='cpu', dtype=torch.float32)
    constant_pad_nd_2 = rand_strided((4, 256, 49, 49), (614656, 1, 12544, 256), device='cpu', dtype=torch.float32)
    squeeze_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    mul_81 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_38 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    squeeze_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 512, 24, 24), (294912, 1, 12288, 512), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((4, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    mul_100 = rand_strided((4, 512, 24, 24), (294912, 1, 12288, 512), device='cpu', dtype=torch.float32)
    squeeze_29 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    mul_107 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    mul_114 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    squeeze_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_50 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    mul_121 = rand_strided((4, 256, 24, 24), (147456, 1, 6144, 256), device='cpu', dtype=torch.float32)
    squeeze_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 512, 24, 24), (294912, 1, 12288, 512), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((4, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    mul_133 = rand_strided((4, 512, 24, 24), (294912, 1, 12288, 512), device='cpu', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((4, 512, 12, 12), (73728, 1, 6144, 512), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_56 = rand_strided((1536, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    squeeze_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((768, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 768, 24, 24), (442368, 1, 18432, 768), device='cpu', dtype=torch.float32)
    constant_pad_nd_3 = rand_strided((4, 768, 25, 25), (480000, 1, 19200, 768), device='cpu', dtype=torch.float32)
    squeeze_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_150 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_157 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_45 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_169 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    squeeze_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_176 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_74 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_183 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_190 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_53 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_80 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_202 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_209 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_216 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_223 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_92 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_235 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    squeeze_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_242 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_98 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_249 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_256 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_69 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_268 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    squeeze_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_275 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_282 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_113 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_289 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_77 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_116 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_301 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_308 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_81 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_122 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_315 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    mul_322 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_334 = rand_strided((4, 1536, 12, 12), (221184, 1, 18432, 1536), device='cpu', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    squeeze_87 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_131 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    squeeze_89 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((4, 768, 12, 12), (110592, 1, 9216, 768), device='cpu', dtype=torch.float32)
    constant_pad_nd_4 = rand_strided((4, 768, 13, 13), (129792, 1, 9984, 768), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_137 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_351 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_358 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_95 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_143 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_370 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_146 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_377 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_149 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_384 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_101 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_391 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_155 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_403 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    squeeze_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_158 = rand_strided((768, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_410 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_161 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_417 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view_164 = rand_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    mul_424 = rand_strided((4, 768, 6, 6), (27648, 1, 4608, 768), device='cpu', dtype=torch.float32)
    squeeze_111 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_167 = rand_strided((1536, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((4, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    add_118 = rand_strided((4, 1536, 6, 6), (55296, 1, 9216, 1536), device='cpu', dtype=torch.float32)
    squeeze_113 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((4, 3072, 6, 6), (110592, 1, 18432, 3072), device='cpu', dtype=torch.float32)
    clone_12 = rand_strided((4, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    unsqueeze_58 = rand_strided((1, 3072, 1), (3072, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_66 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_74 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_82 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_90 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_98 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_106 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_114 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_122 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_146 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_170 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_194 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_218 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 32, 1), (32, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_83, primals_84, primals_86, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_156, primals_158, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, constant_pad_nd, squeeze_1, view_2, convolution, mul_6, squeeze_3, view_5, convolution_1, mul_13, squeeze_5, view_8, convolution_2, constant_pad_nd_1, squeeze_7, view_11, convolution_3, mul_28, squeeze_9, view_14, convolution_4, squeeze_11, view_17, convolution_5, mul_38, squeeze_13, view_20, convolution_6, mul_45, squeeze_15, view_23, convolution_7, mul_52, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_64, avg_pool2d, squeeze_19, view_29, convolution_11, squeeze_21, view_32, convolution_12, constant_pad_nd_2, squeeze_23, view_35, convolution_13, mul_81, squeeze_25, view_38, convolution_14, mul_88, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_100, squeeze_29, view_44, convolution_18, mul_107, squeeze_31, view_47, convolution_19, mul_114, squeeze_33, view_50, convolution_20, mul_121, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_133, avg_pool2d_1, squeeze_37, view_56, convolution_24, squeeze_39, view_59, convolution_25, constant_pad_nd_3, squeeze_41, view_62, convolution_26, mul_150, squeeze_43, view_65, convolution_27, mul_157, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_169, squeeze_47, view_71, convolution_31, mul_176, squeeze_49, view_74, convolution_32, mul_183, squeeze_51, view_77, convolution_33, mul_190, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_202, squeeze_55, view_83, convolution_37, mul_209, squeeze_57, view_86, convolution_38, mul_216, squeeze_59, view_89, convolution_39, mul_223, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_235, squeeze_63, view_95, convolution_43, mul_242, squeeze_65, view_98, convolution_44, mul_249, squeeze_67, view_101, convolution_45, mul_256, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_268, squeeze_71, view_107, convolution_49, mul_275, squeeze_73, view_110, convolution_50, mul_282, squeeze_75, view_113, convolution_51, mul_289, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_301, squeeze_79, view_119, convolution_55, mul_308, squeeze_81, view_122, convolution_56, mul_315, squeeze_83, view_125, convolution_57, mul_322, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_334, avg_pool2d_2, squeeze_87, view_131, convolution_61, squeeze_89, view_134, convolution_62, constant_pad_nd_4, squeeze_91, view_137, convolution_63, mul_351, squeeze_93, view_140, convolution_64, mul_358, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_370, squeeze_97, view_146, convolution_68, mul_377, squeeze_99, view_149, convolution_69, mul_384, squeeze_101, view_152, convolution_70, mul_391, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_403, squeeze_105, view_158, convolution_74, mul_410, squeeze_107, view_161, convolution_75, mul_417, squeeze_109, view_164, convolution_76, mul_424, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_118, squeeze_113, view_170, convolution_80, clone_12, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_nfnet', benchmark_compiled_module)
