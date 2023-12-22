
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


cpp_fused_clone_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_flip_full_like_masked_fill_new_zeros_select_scatter_slice_slice_scatter_tril_view_where_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15)
{
    auto out_ptr12 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>((-255L) + x0 + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        out_ptr1[static_cast<long>(x1 + (257L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = out_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(256L + ((-1L)*x0) + ((-1L)*x1));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        out_ptr4[static_cast<long>(x1 + (257L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr2[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr3[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr5[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(1.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 != tmp2);
                    auto tmp4 = mask_convert_to_float(tmp3);
                    auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp3);
                    tmp7.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(513L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(3);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = c10::convert<long>(x0);
                        auto tmp4 = static_cast<long>(256);
                        auto tmp5 = tmp3 >= tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x0 + (513L*x2)), 512L)) % static_cast<long>(513L));
                            auto tmp8 = static_cast<long>(512);
                            auto tmp9 = tmp7 < tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = out_ptr6[static_cast<long>((256L*(c10::div_floor_integer((656384L + x0 + (513L*x2)), 262656L))) + (static_cast<long>(c10::div_floor_integer((656384L + x0 + (513L*x2)), 512L)) % static_cast<long>(513L)))];
                                auto tmp12 = out_ptr7[static_cast<long>((256L*(c10::div_floor_integer((656384L + x0 + (513L*x2)), 262656L))) + (static_cast<long>((x0 + (513L*x2))) % static_cast<long>(512L)))];
                                auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp16 = static_cast<long>(3);
                        auto tmp17 = tmp16 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = c10::convert<long>(x0);
                            auto tmp20 = static_cast<long>(256);
                            auto tmp21 = tmp19 >= tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x0 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp24 = static_cast<long>(512);
                                auto tmp25 = tmp23 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer((787712L + x0 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer((787712L + x0 + (513L*x2)), 512L)) % static_cast<long>(513L)))];
                                    auto tmp28 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer((787712L + x0 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>((787712L + x0 + (513L*x2))) % static_cast<long>(512L)))];
                                    auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                                    return tmp29;
                                }
                                ;
                                auto tmp30 = tmp25 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                return tmp30;
                            }
                            ;
                            auto tmp31 = tmp21 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                            auto tmp32 = static_cast<float>(0.0);
                            auto tmp33 = tmp21 ? tmp31 : tmp32;
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp17 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp35 = static_cast<float>(0.0);
                        auto tmp36 = tmp17 ? tmp34 : tmp35;
                        auto tmp37 = tmp5 ? tmp15 : tmp36;
                        auto tmp38 = c10::convert<long>(x1);
                        auto tmp39 = tmp38 < tmp16;
                        auto tmp40 = [&]
                        {
                            auto tmp41 = c10::convert<long>(x0);
                            auto tmp42 = static_cast<long>(256);
                            auto tmp43 = tmp41 >= tmp42;
                            auto tmp44 = [&]
                            {
                                auto tmp45 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x0 + (513L*x2) + (262656L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp46 = static_cast<long>(512);
                                auto tmp47 = tmp45 < tmp46;
                                auto tmp48 = [&]
                                {
                                    auto tmp49 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-256L) + x0 + (513L*x2) + (262656L*x1)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer(((-256L) + x0 + (513L*x2) + (262656L*x1)), 512L)) % static_cast<long>(513L)))];
                                    auto tmp50 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-256L) + x0 + (513L*x2) + (262656L*x1)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(((-256L) + x0 + (513L*x2) + (262656L*x1))) % static_cast<long>(512L)))];
                                    auto tmp51 = decltype(tmp49)(tmp49 * tmp50);
                                    return tmp51;
                                }
                                ;
                                auto tmp52 = tmp47 ? tmp48() : static_cast<decltype(tmp48())>(0.0);
                                return tmp52;
                            }
                            ;
                            auto tmp53 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                            auto tmp54 = static_cast<float>(0.0);
                            auto tmp55 = tmp43 ? tmp53 : tmp54;
                            return tmp55;
                        }
                        ;
                        auto tmp56 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                        auto tmp57 = tmp39 ? tmp56 : tmp35;
                        auto tmp58 = tmp2 ? tmp37 : tmp57;
                        out_ptr8[static_cast<long>(x2 + (256L*x1) + (1024L*x0))] = tmp58;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp62 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(1);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1);
                        auto tmp5 = static_cast<long>(1);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(256);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L));
                            auto tmp12 = static_cast<long>(512);
                            auto tmp13 = tmp11 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-256L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer(((-256L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L)))];
                                auto tmp16 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-256L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(((-256L) + x1 + (513L*x0))) % static_cast<long>(512L)))];
                                auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                return tmp17;
                            }
                            ;
                            auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        auto tmp20 = static_cast<long>(0);
                        auto tmp21 = tmp20 >= tmp5;
                        auto tmp22 = [&]
                        {
                            auto tmp23 = c10::convert<long>(x1);
                            auto tmp24 = static_cast<long>(256);
                            auto tmp25 = tmp23 < tmp24;
                            auto tmp26 = [&]
                            {
                                auto tmp27 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp28 = static_cast<long>(512);
                                auto tmp29 = tmp27 < tmp28;
                                auto tmp30 = [&]
                                {
                                    auto tmp31 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L)))];
                                    auto tmp32 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>((x1 + (513L*x0))) % static_cast<long>(512L)))];
                                    auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                    return tmp33;
                                }
                                ;
                                auto tmp34 = tmp29 ? tmp30() : static_cast<decltype(tmp30())>(0.0);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp25 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                            auto tmp36 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                            auto tmp37 = tmp25 ? tmp35 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp21 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                        auto tmp39 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                        auto tmp40 = tmp21 ? tmp38 : tmp39;
                        auto tmp41 = tmp9 ? tmp19 : tmp40;
                        return tmp41;
                    }
                    ;
                    auto tmp42 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp43 = static_cast<long>(0);
                    auto tmp44 = tmp43 >= tmp1;
                    auto tmp45 = [&]
                    {
                        auto tmp46 = c10::convert<long>(x1);
                        auto tmp47 = static_cast<long>(256);
                        auto tmp48 = tmp46 < tmp47;
                        auto tmp49 = [&]
                        {
                            auto tmp50 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L));
                            auto tmp51 = static_cast<long>(512);
                            auto tmp52 = tmp50 < tmp51;
                            auto tmp53 = [&]
                            {
                                auto tmp54 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L)))];
                                auto tmp55 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>((x1 + (513L*x0))) % static_cast<long>(512L)))];
                                auto tmp56 = decltype(tmp54)(tmp54 * tmp55);
                                return tmp56;
                            }
                            ;
                            auto tmp57 = tmp52 ? tmp53() : static_cast<decltype(tmp53())>(0.0);
                            return tmp57;
                        }
                        ;
                        auto tmp58 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                        auto tmp59 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                        auto tmp60 = tmp48 ? tmp58 : tmp59;
                        return tmp60;
                    }
                    ;
                    auto tmp61 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                    auto tmp63 = tmp44 ? tmp61 : tmp62;
                    auto tmp64 = tmp2 ? tmp42 : tmp63;
                    out_ptr9[static_cast<long>(x1 + (513L*x0))] = tmp64;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr9[static_cast<long>(x1 + (513L*x0))];
                        auto tmp29 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                        auto tmp0 = c10::convert<long>((-255L) + x0 + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp2 ? tmp3 : tmp4;
                        auto tmp6 = c10::convert<bool>(tmp5);
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp7 == tmp7;
                        auto tmp10 = static_cast<long>(1);
                        auto tmp11 = tmp1 >= tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = c10::convert<long>(x1);
                            auto tmp14 = static_cast<long>(256);
                            auto tmp15 = tmp13 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp18 = static_cast<long>(512);
                                auto tmp19 = tmp17 < tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 512L)) % static_cast<long>(513L)))];
                                    auto tmp22 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*x0)), 262656L)) % static_cast<long>(3L))) + (static_cast<long>((x1 + (513L*x0))) % static_cast<long>(512L)))];
                                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp26 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                            auto tmp27 = tmp15 ? tmp25 : tmp26;
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp30 = tmp11 ? tmp28 : tmp29;
                        auto tmp31 = tmp8 ? tmp9 : tmp30;
                        auto tmp32 = -std::numeric_limits<float>::infinity();
                        auto tmp33 = tmp6 ? tmp32 : tmp31;
                        out_ptr10[static_cast<long>(x1 + (257L*x0))] = tmp33;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp42 = out_ptr9[static_cast<long>(x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))];
                    auto tmp63 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(256);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1);
                        auto tmp5 = static_cast<long>(257);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = out_ptr10[static_cast<long>(x1 + (257L*x0))];
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                        auto tmp10 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<int>(0);
                        auto tmp12 = tmp10 == tmp11;
                        auto tmp13 = out_ptr9[static_cast<long>(x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))];
                        auto tmp14 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp15 = static_cast<long>(1);
                        auto tmp16 = tmp14 >= tmp15;
                        auto tmp17 = [&]
                        {
                            auto tmp18 = c10::convert<long>(x1);
                            auto tmp19 = static_cast<long>(256);
                            auto tmp20 = tmp18 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 512L)) % static_cast<long>(513L));
                                auto tmp23 = static_cast<long>(512);
                                auto tmp24 = tmp22 < tmp23;
                                auto tmp25 = [&]
                                {
                                    auto tmp26 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 512L)) % static_cast<long>(513L)))];
                                    auto tmp27 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 262656L)) % static_cast<long>(3L))) + (static_cast<long>((x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp24 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp31 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                            auto tmp32 = tmp20 ? tmp30 : tmp31;
                            return tmp32;
                        }
                        ;
                        auto tmp33 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp34 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                        auto tmp35 = tmp16 ? tmp33 : tmp34;
                        auto tmp36 = tmp12 ? tmp13 : tmp35;
                        auto tmp37 = tmp6 ? tmp9 : tmp36;
                        return tmp37;
                    }
                    ;
                    auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp39 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                    auto tmp40 = static_cast<int>(0);
                    auto tmp41 = tmp39 == tmp40;
                    auto tmp43 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                    auto tmp44 = static_cast<long>(1);
                    auto tmp45 = tmp43 >= tmp44;
                    auto tmp46 = [&]
                    {
                        auto tmp47 = c10::convert<long>(x1);
                        auto tmp48 = static_cast<long>(256);
                        auto tmp49 = tmp47 < tmp48;
                        auto tmp50 = [&]
                        {
                            auto tmp51 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 512L)) % static_cast<long>(513L));
                            auto tmp52 = static_cast<long>(512);
                            auto tmp53 = tmp51 < tmp52;
                            auto tmp54 = [&]
                            {
                                auto tmp55 = out_ptr6[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 262656L)) % static_cast<long>(3L))) + (static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 512L)) % static_cast<long>(513L)))];
                                auto tmp56 = out_ptr7[static_cast<long>((256L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L)))), 262656L)) % static_cast<long>(3L))) + (static_cast<long>((x1 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp53 ? tmp54() : static_cast<decltype(tmp54())>(0.0);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp49 ? tmp50() : static_cast<decltype(tmp50())>(0.0);
                        auto tmp60 = out_ptr8[static_cast<long>(x0 + (1024L*x1))];
                        auto tmp61 = tmp49 ? tmp59 : tmp60;
                        return tmp61;
                    }
                    ;
                    auto tmp62 = tmp45 ? tmp46() : static_cast<decltype(tmp46())>(0.0);
                    auto tmp64 = tmp45 ? tmp62 : tmp63;
                    auto tmp65 = tmp41 ? tmp42 : tmp64;
                    auto tmp66 = tmp2 ? tmp38 : tmp65;
                    out_ptr11[static_cast<long>(x1 + (513L*x0))] = tmp66;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr5[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = out_ptr11[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = out_ptr4[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr5[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr5[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = out_ptr11[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = out_ptr11[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr12[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr12 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr12[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr13[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr12 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr13[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr12[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr13[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr14[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr14[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr15 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr14[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr15[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (768L*x3) + (196608L*x1)), static_cast<long>(768L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (32768L*x1) + (98304L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(3);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = static_cast<long>(256);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                auto tmp8 = static_cast<long>(512);
                                auto tmp9 = tmp7 < tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer((656384L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L))) + (262144L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 262656L))) + (786432L*x0) + (786432L*(c10::div_floor_integer((656384L + x3 + (513L*x2)), 787968L))) + (static_cast<long>((x3 + (513L*x2))) % static_cast<long>(512L)))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp14 = static_cast<long>(3);
                            auto tmp15 = tmp14 < tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = c10::convert<long>(x3);
                                auto tmp18 = static_cast<long>(256);
                                auto tmp19 = tmp17 >= tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 512L)) % static_cast<long>(513L));
                                    auto tmp22 = static_cast<long>(512);
                                    auto tmp23 = tmp21 < tmp22;
                                    auto tmp24 = [&]
                                    {
                                        auto tmp25 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2)), 262656L)) % static_cast<long>(3L))) + (786432L*(static_cast<long>(c10::div_floor_integer((787712L + x3 + (513L*x2) + (787968L*x0)), 787968L)) % static_cast<long>(12L))) + (static_cast<long>((787712L + x3 + (513L*x2))) % static_cast<long>(262656L)))];
                                        return tmp25;
                                    }
                                    ;
                                    auto tmp26 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                    return tmp26;
                                }
                                ;
                                auto tmp27 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp19 ? tmp27 : tmp28;
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp15 ? tmp16() : static_cast<decltype(tmp16())>(0.0);
                            auto tmp31 = static_cast<float>(0.0);
                            auto tmp32 = tmp15 ? tmp30 : tmp31;
                            auto tmp33 = tmp5 ? tmp13 : tmp32;
                            auto tmp34 = c10::convert<long>(x1);
                            auto tmp35 = tmp34 < tmp14;
                            auto tmp36 = [&]
                            {
                                auto tmp37 = c10::convert<long>(x3);
                                auto tmp38 = static_cast<long>(256);
                                auto tmp39 = tmp37 >= tmp38;
                                auto tmp40 = [&]
                                {
                                    auto tmp41 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp42 = static_cast<long>(512);
                                    auto tmp43 = tmp41 < tmp42;
                                    auto tmp44 = [&]
                                    {
                                        auto tmp45 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x3 + (513L*x2) + (262656L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                        return tmp45;
                                    }
                                    ;
                                    auto tmp46 = tmp43 ? tmp44() : static_cast<decltype(tmp44())>(0.0);
                                    return tmp46;
                                }
                                ;
                                auto tmp47 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                                auto tmp48 = static_cast<float>(0.0);
                                auto tmp49 = tmp39 ? tmp47 : tmp48;
                                return tmp49;
                            }
                            ;
                            auto tmp50 = tmp35 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp51 = tmp35 ? tmp50 : tmp31;
                            auto tmp52 = tmp2 ? tmp33 : tmp51;
                            out_ptr0[static_cast<long>(x3 + (513L*x2) + (131328L*x1) + (525312L*x0))] = tmp52;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp56 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(1);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = static_cast<long>(256);
                            auto tmp8 = tmp4 < tmp7;
                            auto tmp9 = tmp6 & tmp8;
                            auto tmp10 = [&]
                            {
                                auto tmp11 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp12 = static_cast<long>(512);
                                auto tmp13 = tmp11 < tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr0[static_cast<long>((262144L*(static_cast<long>(c10::div_floor_integer(((-256L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>(((-256L) + x2 + (513L*x1) + (787968L*x0))) % static_cast<long>(262656L)))];
                                    return tmp15;
                                }
                                ;
                                auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                            auto tmp18 = static_cast<long>(0);
                            auto tmp19 = tmp18 >= tmp5;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x2);
                                auto tmp22 = static_cast<long>(256);
                                auto tmp23 = tmp21 < tmp22;
                                auto tmp24 = [&]
                                {
                                    auto tmp25 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp26 = static_cast<long>(512);
                                    auto tmp27 = tmp25 < tmp26;
                                    auto tmp28 = [&]
                                    {
                                        auto tmp29 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp29;
                                    }
                                    ;
                                    auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                                    return tmp30;
                                }
                                ;
                                auto tmp31 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                                auto tmp32 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp33 = tmp23 ? tmp31 : tmp32;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp35 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp36 = tmp19 ? tmp34 : tmp35;
                            auto tmp37 = tmp9 ? tmp17 : tmp36;
                            return tmp37;
                        }
                        ;
                        auto tmp38 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp39 = static_cast<long>(0);
                        auto tmp40 = tmp39 >= tmp1;
                        auto tmp41 = [&]
                        {
                            auto tmp42 = c10::convert<long>(x2);
                            auto tmp43 = static_cast<long>(256);
                            auto tmp44 = tmp42 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp47 = static_cast<long>(512);
                                auto tmp48 = tmp46 < tmp47;
                                auto tmp49 = [&]
                                {
                                    auto tmp50 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp50;
                                }
                                ;
                                auto tmp51 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                                return tmp51;
                            }
                            ;
                            auto tmp52 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp53 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp54 = tmp44 ? tmp52 : tmp53;
                            return tmp54;
                        }
                        ;
                        auto tmp55 = tmp40 ? tmp41() : static_cast<decltype(tmp41())>(0.0);
                        auto tmp57 = tmp40 ? tmp55 : tmp56;
                        auto tmp58 = tmp2 ? tmp38 : tmp57;
                        auto tmp59 = c10::convert<long>(x2);
                        auto tmp60 = static_cast<long>(257);
                        auto tmp61 = tmp59 < tmp60;
                        auto tmp62 = [&]
                        {
                            auto tmp63 = in_ptr1[static_cast<long>(x2 + (257L*x1))];
                            auto tmp64 = c10::convert<bool>(tmp63);
                            auto tmp65 = static_cast<int>(0);
                            auto tmp66 = tmp65 == tmp65;
                            auto tmp67 = static_cast<long>(0);
                            auto tmp68 = static_cast<long>(1);
                            auto tmp69 = tmp67 >= tmp68;
                            auto tmp70 = [&]
                            {
                                auto tmp71 = c10::convert<long>(x2);
                                auto tmp72 = static_cast<long>(256);
                                auto tmp73 = tmp71 < tmp72;
                                auto tmp74 = [&]
                                {
                                    auto tmp75 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                    auto tmp76 = static_cast<long>(512);
                                    auto tmp77 = tmp75 < tmp76;
                                    auto tmp78 = [&]
                                    {
                                        auto tmp79 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                        return tmp79;
                                    }
                                    ;
                                    auto tmp80 = tmp77 ? tmp78() : static_cast<decltype(tmp78())>(0.0);
                                    return tmp80;
                                }
                                ;
                                auto tmp81 = tmp73 ? tmp74() : static_cast<decltype(tmp74())>(0.0);
                                auto tmp82 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp83 = tmp73 ? tmp81 : tmp82;
                                return tmp83;
                            }
                            ;
                            auto tmp84 = tmp69 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                            auto tmp85 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp86 = tmp69 ? tmp84 : tmp85;
                            auto tmp87 = tmp66 ? tmp58 : tmp86;
                            auto tmp88 = -std::numeric_limits<float>::infinity();
                            auto tmp89 = tmp64 ? tmp88 : tmp87;
                            return tmp89;
                        }
                        ;
                        auto tmp90 = tmp61 ? tmp62() : static_cast<decltype(tmp62())>(0.0);
                        auto tmp91 = static_cast<int>(0);
                        auto tmp92 = tmp91 == tmp91;
                        auto tmp93 = [&]
                        {
                            auto tmp94 = c10::convert<long>(x2);
                            auto tmp95 = static_cast<long>(256);
                            auto tmp96 = tmp94 < tmp95;
                            auto tmp97 = [&]
                            {
                                auto tmp98 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L));
                                auto tmp99 = static_cast<long>(512);
                                auto tmp100 = tmp98 < tmp99;
                                auto tmp101 = [&]
                                {
                                    auto tmp102 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*x1) + (787968L*x0)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*x1))) % static_cast<long>(512L)))];
                                    return tmp102;
                                }
                                ;
                                auto tmp103 = tmp100 ? tmp101() : static_cast<decltype(tmp101())>(0.0);
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp96 ? tmp97() : static_cast<decltype(tmp97())>(0.0);
                            auto tmp105 = out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                            auto tmp106 = tmp96 ? tmp104 : tmp105;
                            return tmp106;
                        }
                        ;
                        auto tmp107 = tmp40 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                        auto tmp108 = tmp40 ? tmp107 : tmp56;
                        auto tmp109 = tmp92 ? tmp58 : tmp108;
                        auto tmp110 = tmp61 ? tmp90 : tmp109;
                        out_ptr1[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp58;
                        out_ptr2[static_cast<long>(x2 + (513L*x1) + (131328L*x0))] = tmp110;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp9 = out_ptr1[static_cast<long>(x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (131328L*x1))];
                        auto tmp28 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(x2 + (513L*x0) + (131328L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = c10::convert<int>(c10::div_floor_integer(x0, 256L));
                        auto tmp7 = static_cast<int>(0);
                        auto tmp8 = tmp6 == tmp7;
                        auto tmp10 = c10::convert<long>(c10::div_floor_integer(x0, 256L));
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp10 >= tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = c10::convert<long>(x2);
                            auto tmp15 = static_cast<long>(256);
                            auto tmp16 = tmp14 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = c10::convert<long>(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L));
                                auto tmp19 = static_cast<long>(512);
                                auto tmp20 = tmp18 < tmp19;
                                auto tmp21 = [&]
                                {
                                    auto tmp22 = in_ptr0[static_cast<long>((512L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 512L)) % static_cast<long>(513L))) + (262144L*(static_cast<long>(c10::div_floor_integer(((-131584L) + x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))) + (262656L*(c10::div_floor_integer(x0, 256L))) + (787968L*x1)), 262656L)) % static_cast<long>(36L))) + (static_cast<long>((x2 + (513L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(512L)))];
                                    return tmp22;
                                }
                                ;
                                auto tmp23 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp16 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp25 = out_ptr0[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                            auto tmp26 = tmp16 ? tmp24 : tmp25;
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp29 = tmp12 ? tmp27 : tmp28;
                        auto tmp30 = tmp8 ? tmp9 : tmp29;
                        auto tmp31 = tmp2 ? tmp5 : tmp30;
                        out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp31;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                        auto tmp38 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr2[static_cast<long>((-197632L) + x2 + (257L*x0))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                                auto tmp11 = -std::numeric_limits<float>::infinity();
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = out_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*(static_cast<long>(x0) % static_cast<long>(256L))) + (1575936L*(static_cast<long>(c10::div_floor_integer(((256L*(c10::div_floor_integer(x0, 256L))) + (static_cast<long>(x0) % static_cast<long>(256L))), 256L)) % static_cast<long>(4L))))];
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = c10::convert<long>(x2);
                            auto tmp21 = static_cast<long>(256);
                            auto tmp22 = tmp20 >= tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(1280L + ((-1L)*x0) + ((-1L)*x2));
                                auto tmp25 = static_cast<long>(0);
                                auto tmp26 = tmp24 <= tmp25;
                                auto tmp27 = static_cast<float>(1.0);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = tmp26 ? tmp27 : tmp28;
                                auto tmp30 = c10::convert<bool>(tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                                auto tmp32 = -std::numeric_limits<float>::infinity();
                                auto tmp33 = tmp30 ? tmp32 : tmp31;
                                return tmp33;
                            }
                            ;
                            auto tmp34 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp35 = in_ptr3[static_cast<long>(x2 + (513L*x0))];
                            auto tmp36 = tmp22 ? tmp34 : tmp35;
                            return tmp36;
                        }
                        ;
                        auto tmp37 = tmp2 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp39 = tmp2 ? tmp37 : tmp38;
                        auto tmp40 = decltype(tmp18)(tmp18 + tmp39);
                        out_ptr4[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp40;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (513L*x0)));
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (513L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr4[static_cast<long>(x1 + (513L*x0))];
                        auto tmp1 = out_ptr5[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (513L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = to_float_mask(tmp0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp5);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp6);
                        tmp8.store(out_ptr7 + static_cast<long>(x2 + (513L*x1) + (6156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp2 = out_ptr6[static_cast<long>(x1 + (12L*x0))];
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp0 ? tmp4 : tmp3;
                        out_ptr7[static_cast<long>(x2 + (513L*x1) + (6156L*x0))] = tmp5;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_constant_pad_nd_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>((-256L) + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1024);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = tmp2 & tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr0 + static_cast<long>((-196608L) + x2 + (64L*x0) + (768L*x1)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(-1.0)), tmp6(), to_float_mask(tmp5));
                        tmp8.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(770L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(513);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr1[static_cast<long>(x2 + (513L*x0) + (6156L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x2 + (770L*x1) + (788480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16384L*x1) + (98304L*x0)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (49152L*x1) + (196608L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_add_detach_native_layer_norm_native_layer_norm_backward_slice_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       float* in_out_ptr6,
                       float* in_out_ptr7,
                       float* in_out_ptr8,
                       float* in_out_ptr9,
                       float* in_out_ptr10,
                       float* in_out_ptr11,
                       float* in_out_ptr12,
                       float* in_out_ptr13,
                       float* in_out_ptr14,
                       float* in_out_ptr15,
                       float* in_out_ptr16,
                       float* in_out_ptr17,
                       float* in_out_ptr18,
                       float* in_out_ptr19,
                       float* in_out_ptr20,
                       float* in_out_ptr21,
                       float* in_out_ptr22,
                       float* in_out_ptr23,
                       float* in_out_ptr24,
                       float* in_out_ptr25,
                       float* in_out_ptr26,
                       float* in_out_ptr27,
                       float* in_out_ptr28,
                       float* in_out_ptr29,
                       float* in_out_ptr30,
                       float* in_out_ptr31,
                       float* in_out_ptr32,
                       float* in_out_ptr33,
                       float* in_out_ptr34,
                       float* in_out_ptr35,
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr2 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr2[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr2[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr5 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr5[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr5[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr8 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr8[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr8[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr11 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr11[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr11[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr12 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr14 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr14[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr14[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr11[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr17 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr17[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr11[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr17[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr20 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr20[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr20[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr22 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr13[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr23 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr23[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr13[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr23[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr14[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr26 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr26[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr14[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr26[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr15[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr29 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr29[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr15[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr29[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr16[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr32 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr32[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr16[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr32[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr33 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr34 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x1 + (513L*x0)));
                    auto tmp1 = in_ptr17[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr35 + static_cast<long>(x1 + (513L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(512L); x1<static_cast<long>(513L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr35[static_cast<long>(x1 + (513L*x0))];
                    auto tmp1 = in_ptr17[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr35[static_cast<long>(x1 + (513L*x0))] = tmp2;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195 = args
    args.clear()
    assert_size_stride(primals_1, (768, 768), (768, 1))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, 768), (768, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 768), (768, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, 768), (768, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (3072, 768), (768, 1))
    assert_size_stride(primals_12, (3072, ), (1, ))
    assert_size_stride(primals_13, (768, 3072), (3072, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, 768), (768, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, 768), (768, 1))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 768), (768, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, 768), (768, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (3072, 768), (768, 1))
    assert_size_stride(primals_28, (3072, ), (1, ))
    assert_size_stride(primals_29, (768, 3072), (3072, 1))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, 768), (768, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, 768), (768, 1))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 768), (768, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, 768), (768, 1))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (3072, 768), (768, 1))
    assert_size_stride(primals_44, (3072, ), (1, ))
    assert_size_stride(primals_45, (768, 3072), (3072, 1))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, 768), (768, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, 768), (768, 1))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 768), (768, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, 768), (768, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (3072, 768), (768, 1))
    assert_size_stride(primals_60, (3072, ), (1, ))
    assert_size_stride(primals_61, (768, 3072), (3072, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768), (768, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, 768), (768, 1))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, 768), (768, 1))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (3072, 768), (768, 1))
    assert_size_stride(primals_76, (3072, ), (1, ))
    assert_size_stride(primals_77, (768, 3072), (3072, 1))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768), (768, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, 768), (768, 1))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 768), (768, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, 768), (768, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (3072, 768), (768, 1))
    assert_size_stride(primals_92, (3072, ), (1, ))
    assert_size_stride(primals_93, (768, 3072), (3072, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, 768), (768, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 768), (768, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, 768), (768, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (3072, 768), (768, 1))
    assert_size_stride(primals_108, (3072, ), (1, ))
    assert_size_stride(primals_109, (768, 3072), (3072, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768), (768, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, 768), (768, 1))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, 768), (768, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (3072, 768), (768, 1))
    assert_size_stride(primals_124, (3072, ), (1, ))
    assert_size_stride(primals_125, (768, 3072), (3072, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, 768), (768, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, 768), (768, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, 768), (768, 1))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (3072, 768), (768, 1))
    assert_size_stride(primals_140, (3072, ), (1, ))
    assert_size_stride(primals_141, (768, 3072), (3072, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 768), (768, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, 768), (768, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 768), (768, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, 768), (768, 1))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (3072, 768), (768, 1))
    assert_size_stride(primals_156, (3072, ), (1, ))
    assert_size_stride(primals_157, (768, 3072), (3072, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768), (768, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, 768), (768, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 768), (768, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, 768), (768, 1))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (3072, 768), (768, 1))
    assert_size_stride(primals_172, (3072, ), (1, ))
    assert_size_stride(primals_173, (768, 3072), (3072, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768), (768, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, 768), (768, 1))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 768), (768, 1))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, 768), (768, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (3072, 768), (768, 1))
    assert_size_stride(primals_188, (3072, ), (1, ))
    assert_size_stride(primals_189, (768, 3072), (3072, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(primals_194, (1, 1024), (1024, 1))
    assert_size_stride(primals_195, (1, 1024), (1024, 1))
    buf0 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_vectors], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_2, reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), reinterpret_tensor(primals_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf0)
    del primals_2
    buf1 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_4, reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), reinterpret_tensor(primals_3, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1)
    del primals_4
    buf2 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), reinterpret_tensor(primals_5, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf2)
    del primals_6
    buf3 = reinterpret_tensor(buf0, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf0  # reuse
    buf4 = empty((12, 3, 512, 64, 1), device='cpu', dtype=torch.float32)
    buf5 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_0(c_void_p(buf3.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    buf6 = empty((36, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf5, (36, 64, 512), (32768, 512, 1), 0), out=buf6)
    buf7 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 256, 1, 257), device='cpu', dtype=torch.float32)
    buf8 = empty((12, 256, 513), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((1, 256, 12, 513), (1575936, 513, 131328, 1), device='cpu', dtype=torch.float32)
    buf10 = empty((1, 256, 1, 257), device='cpu', dtype=torch.float32)
    buf12 = empty((1, 1024, 12, 513), device='cpu', dtype=torch.float32)
    buf13 = empty((1, 2, 512, 1), device='cpu', dtype=torch.float32)
    buf14 = empty((1, 2, 512, 1), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((1, 4, 256, 513), (525312, 256, 1, 1024), device='cpu', dtype=torch.float32)
    buf16 = empty((1, 256, 513), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((1, 256, 1, 257), (65792, 257, 65792, 1), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((1, 1024, 1, 513), (525312, 513, 525312, 1), device='cpu', dtype=torch.float32)
    buf19 = empty((1, 1024, 12, 513), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf21 = buf19; del buf19  # reuse
    buf22 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf23 = empty((1, 1024, 12, 513), device='cpu', dtype=torch.float32)
    cpp_fused__softmax__to_copy_add_copy_flip_full_like_masked_fill_new_zeros_select_scatter_slice_slice_scatter_tril_view_where_1(c_void_p(buf21.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del buf15
    del buf16
    del buf17
    del primals_194
    # Source Nodes: [attn_probs, attn_probs_1, attn_probs_3, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf24 = aten.native_dropout(buf23, 0.1, True)
    buf25 = buf24[0]
    buf26 = buf24[1]
    del buf24
    buf27 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf28 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf29 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_2(c_void_p(buf2.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf2, (48, 256, 64), (16384, 64, 1), 0); del buf2  # reuse
    # Source Nodes: [context], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf28, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf29, (48, 768, 64), (49152, 64, 1), 0), out=buf30)
    buf31 = reinterpret_tensor(buf3, (1024, 768), (768, 1), 0); del buf3  # reuse
    cpp_fused_view_3(c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    buf32 = reinterpret_tensor(buf30, (1024, 768), (768, 1), 0); del buf30  # reuse
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf31, reinterpret_tensor(primals_7, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf32)
    del primals_8
    # Source Nodes: [hidden_states_6], Original ATen: [aten.native_dropout]
    buf33 = aten.native_dropout(reinterpret_tensor(buf32, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = reinterpret_tensor(buf14, (1, 1024, 1), (1024, 1, 1024), 0); del buf14  # reuse
    buf37 = reinterpret_tensor(buf13, (1, 1024, 1), (1024, 1, 1024), 0); del buf13  # reuse
    buf39 = reinterpret_tensor(buf32, (1, 1024, 768), (786432, 768, 1), 0); del buf32  # reuse
    buf40 = buf1; del buf1  # reuse
    cpp_fused_add_native_layer_norm_view_4(c_void_p(buf34.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf40, reinterpret_tensor(primals_11, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf41)
    del primals_12
    buf42 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_5(c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    buf43 = reinterpret_tensor(buf34, (1024, 768), (768, 1), 0); del buf34  # reuse
    # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf42, reinterpret_tensor(primals_13, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf43)
    del primals_14
    # Source Nodes: [hidden_states_11], Original ATen: [aten.native_dropout]
    buf44 = aten.native_dropout(reinterpret_tensor(buf43, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf45 = buf44[0]
    buf46 = buf44[1]
    del buf44
    buf47 = buf36; del buf36  # reuse
    buf48 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf50 = reinterpret_tensor(buf43, (1, 1024, 768), (786432, 768, 1), 0); del buf43  # reuse
    buf51 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_6(c_void_p(buf45.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del primals_10
    buf52 = reinterpret_tensor(buf45, (1024, 768), (768, 1), 0); del buf45  # reuse
    # Source Nodes: [query_vectors_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, buf51, reinterpret_tensor(primals_17, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf52)
    del primals_18
    buf53 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, buf51, reinterpret_tensor(primals_19, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
    del primals_20
    buf54 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, buf51, reinterpret_tensor(primals_21, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf54)
    del primals_22
    buf55 = reinterpret_tensor(buf52, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf52  # reuse
    buf56 = reinterpret_tensor(buf27, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf27  # reuse
    buf57 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_7(c_void_p(buf55.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    buf58 = buf6; del buf6  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf56, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf57, (36, 64, 512), (32768, 512, 1), 0), out=buf58)
    buf59 = reinterpret_tensor(buf25, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf25  # reuse
    buf60 = buf8; del buf8  # reuse
    buf61 = buf11; del buf11  # reuse
    buf62 = buf23; del buf23  # reuse
    buf63 = reinterpret_tensor(buf7, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf7  # reuse
    buf64 = buf20; del buf20  # reuse
    buf65 = buf63; del buf63  # reuse
    buf66 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf67 = buf12; del buf12  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_8(c_void_p(buf65.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    # Source Nodes: [attn_probs_4, attn_probs_5, attn_probs_7, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf68 = aten.native_dropout(buf67, 0.1, True)
    buf69 = buf68[0]
    buf70 = buf68[1]
    del buf68
    buf71 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf72 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf73 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_9(c_void_p(buf54.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    buf74 = reinterpret_tensor(buf54, (48, 256, 64), (16384, 64, 1), 0); del buf54  # reuse
    # Source Nodes: [context_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf72, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf73, (48, 768, 64), (49152, 64, 1), 0), out=buf74)
    buf75 = reinterpret_tensor(buf55, (1024, 768), (768, 1), 0); del buf55  # reuse
    cpp_fused_view_10(c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = reinterpret_tensor(buf74, (1024, 768), (768, 1), 0); del buf74  # reuse
    # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf75, reinterpret_tensor(primals_23, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf76)
    del primals_24
    # Source Nodes: [hidden_states_20], Original ATen: [aten.native_dropout]
    buf77 = aten.native_dropout(reinterpret_tensor(buf76, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf80 = buf47; del buf47  # reuse
    buf81 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf83 = reinterpret_tensor(buf76, (1, 1024, 768), (786432, 768, 1), 0); del buf76  # reuse
    buf84 = buf53; del buf53  # reuse
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf78.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    del primals_16
    buf85 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_28, buf84, reinterpret_tensor(primals_27, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf85)
    del primals_28
    buf86 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_12(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    buf87 = reinterpret_tensor(buf78, (1024, 768), (768, 1), 0); del buf78  # reuse
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_30, buf86, reinterpret_tensor(primals_29, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf87)
    del primals_30
    # Source Nodes: [hidden_states_25], Original ATen: [aten.native_dropout]
    buf88 = aten.native_dropout(reinterpret_tensor(buf87, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf89 = buf88[0]
    buf90 = buf88[1]
    del buf88
    buf91 = buf80; del buf80  # reuse
    buf92 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf94 = reinterpret_tensor(buf87, (1, 1024, 768), (786432, 768, 1), 0); del buf87  # reuse
    buf95 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_13(c_void_p(buf89.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del primals_26
    buf96 = reinterpret_tensor(buf89, (1024, 768), (768, 1), 0); del buf89  # reuse
    # Source Nodes: [query_vectors_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_34, buf95, reinterpret_tensor(primals_33, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf96)
    del primals_34
    buf97 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_36, buf95, reinterpret_tensor(primals_35, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf97)
    del primals_36
    buf98 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_38, buf95, reinterpret_tensor(primals_37, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf98)
    del primals_38
    buf99 = reinterpret_tensor(buf96, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf96  # reuse
    buf100 = reinterpret_tensor(buf71, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf71  # reuse
    buf101 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_14(c_void_p(buf99.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    buf102 = buf58; del buf58  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf100, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf101, (36, 64, 512), (32768, 512, 1), 0), out=buf102)
    buf103 = reinterpret_tensor(buf69, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf69  # reuse
    buf104 = reinterpret_tensor(buf61, (12, 256, 513), (131328, 513, 1), 0); del buf61  # reuse
    buf105 = reinterpret_tensor(buf60, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf60  # reuse
    buf106 = buf67; del buf67  # reuse
    buf107 = buf62; del buf62  # reuse
    buf108 = buf64; del buf64  # reuse
    buf109 = buf107; del buf107  # reuse
    buf110 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf111 = reinterpret_tensor(buf59, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf59  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_15(c_void_p(buf109.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    # Source Nodes: [attn_probs_11, attn_probs_8, attn_probs_9, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf112 = aten.native_dropout(buf111, 0.1, True)
    buf113 = buf112[0]
    buf114 = buf112[1]
    del buf112
    buf115 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf116 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf117 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_16(c_void_p(buf98.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = reinterpret_tensor(buf98, (48, 256, 64), (16384, 64, 1), 0); del buf98  # reuse
    # Source Nodes: [context_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf116, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf117, (48, 768, 64), (49152, 64, 1), 0), out=buf118)
    buf119 = reinterpret_tensor(buf99, (1024, 768), (768, 1), 0); del buf99  # reuse
    cpp_fused_view_17(c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = reinterpret_tensor(buf118, (1024, 768), (768, 1), 0); del buf118  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_40, buf119, reinterpret_tensor(primals_39, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf120)
    del primals_40
    # Source Nodes: [hidden_states_34], Original ATen: [aten.native_dropout]
    buf121 = aten.native_dropout(reinterpret_tensor(buf120, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf122 = buf121[0]
    buf123 = buf121[1]
    del buf121
    buf124 = buf91; del buf91  # reuse
    buf125 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf127 = reinterpret_tensor(buf120, (1, 1024, 768), (786432, 768, 1), 0); del buf120  # reuse
    buf128 = buf97; del buf97  # reuse
    cpp_fused_add_native_layer_norm_view_18(c_void_p(buf122.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del primals_32
    buf129 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, buf128, reinterpret_tensor(primals_43, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf129)
    del primals_44
    buf130 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_19(c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    buf131 = reinterpret_tensor(buf122, (1024, 768), (768, 1), 0); del buf122  # reuse
    # Source Nodes: [hidden_states_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_46, buf130, reinterpret_tensor(primals_45, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf131)
    del primals_46
    # Source Nodes: [hidden_states_39], Original ATen: [aten.native_dropout]
    buf132 = aten.native_dropout(reinterpret_tensor(buf131, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf133 = buf132[0]
    buf134 = buf132[1]
    del buf132
    buf135 = buf124; del buf124  # reuse
    buf136 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf138 = reinterpret_tensor(buf131, (1, 1024, 768), (786432, 768, 1), 0); del buf131  # reuse
    buf139 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_20(c_void_p(buf133.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_42
    buf140 = reinterpret_tensor(buf133, (1024, 768), (768, 1), 0); del buf133  # reuse
    # Source Nodes: [query_vectors_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, buf139, reinterpret_tensor(primals_49, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf140)
    del primals_50
    buf141 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_52, buf139, reinterpret_tensor(primals_51, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf141)
    del primals_52
    buf142 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf139, reinterpret_tensor(primals_53, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf142)
    del primals_54
    buf143 = reinterpret_tensor(buf140, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf140  # reuse
    buf144 = reinterpret_tensor(buf115, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf115  # reuse
    buf145 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_21(c_void_p(buf143.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = buf102; del buf102  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf144, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf145, (36, 64, 512), (32768, 512, 1), 0), out=buf146)
    buf147 = reinterpret_tensor(buf113, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf113  # reuse
    buf148 = reinterpret_tensor(buf105, (12, 256, 513), (131328, 513, 1), 0); del buf105  # reuse
    buf149 = reinterpret_tensor(buf104, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf104  # reuse
    buf150 = buf111; del buf111  # reuse
    buf151 = buf106; del buf106  # reuse
    buf152 = buf108; del buf108  # reuse
    buf153 = buf151; del buf151  # reuse
    buf154 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf155 = reinterpret_tensor(buf103, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf103  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_22(c_void_p(buf153.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    # Source Nodes: [attn_probs_12, attn_probs_13, attn_probs_15, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf156 = aten.native_dropout(buf155, 0.1, True)
    buf157 = buf156[0]
    buf158 = buf156[1]
    del buf156
    buf159 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf160 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf161 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_23(c_void_p(buf142.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = reinterpret_tensor(buf142, (48, 256, 64), (16384, 64, 1), 0); del buf142  # reuse
    # Source Nodes: [context_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf160, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf161, (48, 768, 64), (49152, 64, 1), 0), out=buf162)
    buf163 = reinterpret_tensor(buf143, (1024, 768), (768, 1), 0); del buf143  # reuse
    cpp_fused_view_24(c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    buf164 = reinterpret_tensor(buf162, (1024, 768), (768, 1), 0); del buf162  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf163, reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf164)
    del primals_56
    # Source Nodes: [hidden_states_48], Original ATen: [aten.native_dropout]
    buf165 = aten.native_dropout(reinterpret_tensor(buf164, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf166 = buf165[0]
    buf167 = buf165[1]
    del buf165
    buf168 = buf135; del buf135  # reuse
    buf169 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf164, (1, 1024, 768), (786432, 768, 1), 0); del buf164  # reuse
    buf172 = buf141; del buf141  # reuse
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf166.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del primals_48
    buf173 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf172, reinterpret_tensor(primals_59, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf173)
    del primals_60
    buf174 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_26(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = reinterpret_tensor(buf166, (1024, 768), (768, 1), 0); del buf166  # reuse
    # Source Nodes: [hidden_states_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_62, buf174, reinterpret_tensor(primals_61, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf175)
    del primals_62
    # Source Nodes: [hidden_states_53], Original ATen: [aten.native_dropout]
    buf176 = aten.native_dropout(reinterpret_tensor(buf175, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf177 = buf176[0]
    buf178 = buf176[1]
    del buf176
    buf179 = buf168; del buf168  # reuse
    buf180 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf182 = reinterpret_tensor(buf175, (1, 1024, 768), (786432, 768, 1), 0); del buf175  # reuse
    buf183 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_27(c_void_p(buf177.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    del primals_58
    buf184 = reinterpret_tensor(buf177, (1024, 768), (768, 1), 0); del buf177  # reuse
    # Source Nodes: [query_vectors_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_66, buf183, reinterpret_tensor(primals_65, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf184)
    del primals_66
    buf185 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_68, buf183, reinterpret_tensor(primals_67, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf185)
    del primals_68
    buf186 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_70, buf183, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf186)
    del primals_70
    buf187 = reinterpret_tensor(buf184, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf184  # reuse
    buf188 = reinterpret_tensor(buf159, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf159  # reuse
    buf189 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_28(c_void_p(buf187.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = buf146; del buf146  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf189, (36, 64, 512), (32768, 512, 1), 0), out=buf190)
    buf191 = reinterpret_tensor(buf157, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf157  # reuse
    buf192 = reinterpret_tensor(buf149, (12, 256, 513), (131328, 513, 1), 0); del buf149  # reuse
    buf193 = reinterpret_tensor(buf148, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf148  # reuse
    buf194 = buf155; del buf155  # reuse
    buf195 = buf150; del buf150  # reuse
    buf196 = buf152; del buf152  # reuse
    buf197 = buf195; del buf195  # reuse
    buf198 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf199 = reinterpret_tensor(buf147, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf147  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_29(c_void_p(buf197.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    # Source Nodes: [attn_probs_16, attn_probs_17, attn_probs_19, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf200 = aten.native_dropout(buf199, 0.1, True)
    buf201 = buf200[0]
    buf202 = buf200[1]
    del buf200
    buf203 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf204 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf205 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_30(c_void_p(buf186.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    buf206 = reinterpret_tensor(buf186, (48, 256, 64), (16384, 64, 1), 0); del buf186  # reuse
    # Source Nodes: [context_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf204, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf205, (48, 768, 64), (49152, 64, 1), 0), out=buf206)
    buf207 = reinterpret_tensor(buf187, (1024, 768), (768, 1), 0); del buf187  # reuse
    cpp_fused_view_31(c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    buf208 = reinterpret_tensor(buf206, (1024, 768), (768, 1), 0); del buf206  # reuse
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf207, reinterpret_tensor(primals_71, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf208)
    del primals_72
    # Source Nodes: [hidden_states_62], Original ATen: [aten.native_dropout]
    buf209 = aten.native_dropout(reinterpret_tensor(buf208, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf210 = buf209[0]
    buf211 = buf209[1]
    del buf209
    buf212 = buf179; del buf179  # reuse
    buf213 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf215 = reinterpret_tensor(buf208, (1, 1024, 768), (786432, 768, 1), 0); del buf208  # reuse
    buf216 = buf185; del buf185  # reuse
    cpp_fused_add_native_layer_norm_view_32(c_void_p(buf210.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del primals_64
    buf217 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_76, buf216, reinterpret_tensor(primals_75, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf217)
    del primals_76
    buf218 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_33(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf210, (1024, 768), (768, 1), 0); del buf210  # reuse
    # Source Nodes: [hidden_states_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_78, buf218, reinterpret_tensor(primals_77, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf219)
    del primals_78
    # Source Nodes: [hidden_states_67], Original ATen: [aten.native_dropout]
    buf220 = aten.native_dropout(reinterpret_tensor(buf219, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf221 = buf220[0]
    buf222 = buf220[1]
    del buf220
    buf223 = buf212; del buf212  # reuse
    buf224 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf226 = reinterpret_tensor(buf219, (1, 1024, 768), (786432, 768, 1), 0); del buf219  # reuse
    buf227 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_34(c_void_p(buf221.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    del primals_74
    buf228 = reinterpret_tensor(buf221, (1024, 768), (768, 1), 0); del buf221  # reuse
    # Source Nodes: [query_vectors_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf227, reinterpret_tensor(primals_81, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf228)
    del primals_82
    buf229 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf227, reinterpret_tensor(primals_83, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf229)
    del primals_84
    buf230 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf227, reinterpret_tensor(primals_85, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf230)
    del primals_86
    buf231 = reinterpret_tensor(buf228, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf228  # reuse
    buf232 = reinterpret_tensor(buf203, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf203  # reuse
    buf233 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_35(c_void_p(buf231.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = buf190; del buf190  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf232, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf233, (36, 64, 512), (32768, 512, 1), 0), out=buf234)
    buf235 = reinterpret_tensor(buf201, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf201  # reuse
    buf236 = reinterpret_tensor(buf193, (12, 256, 513), (131328, 513, 1), 0); del buf193  # reuse
    buf237 = reinterpret_tensor(buf192, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf192  # reuse
    buf238 = buf199; del buf199  # reuse
    buf239 = buf194; del buf194  # reuse
    buf240 = buf196; del buf196  # reuse
    buf241 = buf239; del buf239  # reuse
    buf242 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf243 = reinterpret_tensor(buf191, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf191  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_36(c_void_p(buf241.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    # Source Nodes: [attn_probs_20, attn_probs_21, attn_probs_23, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf244 = aten.native_dropout(buf243, 0.1, True)
    buf245 = buf244[0]
    buf246 = buf244[1]
    del buf244
    buf247 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf248 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf249 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_37(c_void_p(buf230.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    buf250 = reinterpret_tensor(buf230, (48, 256, 64), (16384, 64, 1), 0); del buf230  # reuse
    # Source Nodes: [context_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf248, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf249, (48, 768, 64), (49152, 64, 1), 0), out=buf250)
    buf251 = reinterpret_tensor(buf231, (1024, 768), (768, 1), 0); del buf231  # reuse
    cpp_fused_view_38(c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = reinterpret_tensor(buf250, (1024, 768), (768, 1), 0); del buf250  # reuse
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf251, reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf252)
    del primals_88
    # Source Nodes: [hidden_states_76], Original ATen: [aten.native_dropout]
    buf253 = aten.native_dropout(reinterpret_tensor(buf252, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf254 = buf253[0]
    buf255 = buf253[1]
    del buf253
    buf256 = buf223; del buf223  # reuse
    buf257 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf259 = reinterpret_tensor(buf252, (1, 1024, 768), (786432, 768, 1), 0); del buf252  # reuse
    buf260 = buf229; del buf229  # reuse
    cpp_fused_add_native_layer_norm_view_39(c_void_p(buf254.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del primals_80
    buf261 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf260, reinterpret_tensor(primals_91, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf261)
    del primals_92
    buf262 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_40(c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    buf263 = reinterpret_tensor(buf254, (1024, 768), (768, 1), 0); del buf254  # reuse
    # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf262, reinterpret_tensor(primals_93, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf263)
    del primals_94
    # Source Nodes: [hidden_states_81], Original ATen: [aten.native_dropout]
    buf264 = aten.native_dropout(reinterpret_tensor(buf263, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf265 = buf264[0]
    buf266 = buf264[1]
    del buf264
    buf267 = buf256; del buf256  # reuse
    buf268 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf270 = reinterpret_tensor(buf263, (1, 1024, 768), (786432, 768, 1), 0); del buf263  # reuse
    buf271 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_41(c_void_p(buf265.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del primals_90
    buf272 = reinterpret_tensor(buf265, (1024, 768), (768, 1), 0); del buf265  # reuse
    # Source Nodes: [query_vectors_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf271, reinterpret_tensor(primals_97, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf272)
    del primals_98
    buf273 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf271, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf273)
    del primals_100
    buf274 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf271, reinterpret_tensor(primals_101, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf274)
    del primals_102
    buf275 = reinterpret_tensor(buf272, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf272  # reuse
    buf276 = reinterpret_tensor(buf247, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf247  # reuse
    buf277 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_42(c_void_p(buf275.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()))
    buf278 = buf234; del buf234  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf276, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf277, (36, 64, 512), (32768, 512, 1), 0), out=buf278)
    buf279 = reinterpret_tensor(buf245, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf245  # reuse
    buf280 = reinterpret_tensor(buf237, (12, 256, 513), (131328, 513, 1), 0); del buf237  # reuse
    buf281 = reinterpret_tensor(buf236, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf236  # reuse
    buf282 = buf243; del buf243  # reuse
    buf283 = buf238; del buf238  # reuse
    buf284 = buf240; del buf240  # reuse
    buf285 = buf283; del buf283  # reuse
    buf286 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf287 = reinterpret_tensor(buf235, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf235  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_43(c_void_p(buf285.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    # Source Nodes: [attn_probs_24, attn_probs_25, attn_probs_27, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf288 = aten.native_dropout(buf287, 0.1, True)
    buf289 = buf288[0]
    buf290 = buf288[1]
    del buf288
    buf291 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf292 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf293 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_44(c_void_p(buf274.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    buf294 = reinterpret_tensor(buf274, (48, 256, 64), (16384, 64, 1), 0); del buf274  # reuse
    # Source Nodes: [context_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf292, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf293, (48, 768, 64), (49152, 64, 1), 0), out=buf294)
    buf295 = reinterpret_tensor(buf275, (1024, 768), (768, 1), 0); del buf275  # reuse
    cpp_fused_view_45(c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    buf296 = reinterpret_tensor(buf294, (1024, 768), (768, 1), 0); del buf294  # reuse
    # Source Nodes: [hidden_states_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf295, reinterpret_tensor(primals_103, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf296)
    del primals_104
    # Source Nodes: [hidden_states_90], Original ATen: [aten.native_dropout]
    buf297 = aten.native_dropout(reinterpret_tensor(buf296, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf298 = buf297[0]
    buf299 = buf297[1]
    del buf297
    buf300 = buf267; del buf267  # reuse
    buf301 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf303 = reinterpret_tensor(buf296, (1, 1024, 768), (786432, 768, 1), 0); del buf296  # reuse
    buf304 = buf273; del buf273  # reuse
    cpp_fused_add_native_layer_norm_view_46(c_void_p(buf298.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del primals_96
    buf305 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf304, reinterpret_tensor(primals_107, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf305)
    del primals_108
    buf306 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_47(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    buf307 = reinterpret_tensor(buf298, (1024, 768), (768, 1), 0); del buf298  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, buf306, reinterpret_tensor(primals_109, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf307)
    del primals_110
    # Source Nodes: [hidden_states_95], Original ATen: [aten.native_dropout]
    buf308 = aten.native_dropout(reinterpret_tensor(buf307, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf309 = buf308[0]
    buf310 = buf308[1]
    del buf308
    buf311 = buf300; del buf300  # reuse
    buf312 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf314 = reinterpret_tensor(buf307, (1, 1024, 768), (786432, 768, 1), 0); del buf307  # reuse
    buf315 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_48(c_void_p(buf309.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    del primals_106
    buf316 = reinterpret_tensor(buf309, (1024, 768), (768, 1), 0); del buf309  # reuse
    # Source Nodes: [query_vectors_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf315, reinterpret_tensor(primals_113, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf316)
    del primals_114
    buf317 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf315, reinterpret_tensor(primals_115, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf317)
    del primals_116
    buf318 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_118, buf315, reinterpret_tensor(primals_117, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf318)
    del primals_118
    buf319 = reinterpret_tensor(buf316, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf316  # reuse
    buf320 = reinterpret_tensor(buf291, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf291  # reuse
    buf321 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_49(c_void_p(buf319.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    buf322 = buf278; del buf278  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf320, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf321, (36, 64, 512), (32768, 512, 1), 0), out=buf322)
    buf323 = reinterpret_tensor(buf289, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf289  # reuse
    buf324 = reinterpret_tensor(buf281, (12, 256, 513), (131328, 513, 1), 0); del buf281  # reuse
    buf325 = reinterpret_tensor(buf280, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf280  # reuse
    buf326 = buf287; del buf287  # reuse
    buf327 = buf282; del buf282  # reuse
    buf328 = buf284; del buf284  # reuse
    buf329 = buf327; del buf327  # reuse
    buf330 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf331 = reinterpret_tensor(buf279, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf279  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_50(c_void_p(buf329.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    # Source Nodes: [attn_probs_28, attn_probs_29, attn_probs_31, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf332 = aten.native_dropout(buf331, 0.1, True)
    buf333 = buf332[0]
    buf334 = buf332[1]
    del buf332
    buf335 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf336 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf337 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_51(c_void_p(buf318.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    buf338 = reinterpret_tensor(buf318, (48, 256, 64), (16384, 64, 1), 0); del buf318  # reuse
    # Source Nodes: [context_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf337, (48, 768, 64), (49152, 64, 1), 0), out=buf338)
    buf339 = reinterpret_tensor(buf319, (1024, 768), (768, 1), 0); del buf319  # reuse
    cpp_fused_view_52(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf338, (1024, 768), (768, 1), 0); del buf338  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf339, reinterpret_tensor(primals_119, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf340)
    del primals_120
    # Source Nodes: [hidden_states_104], Original ATen: [aten.native_dropout]
    buf341 = aten.native_dropout(reinterpret_tensor(buf340, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf342 = buf341[0]
    buf343 = buf341[1]
    del buf341
    buf344 = buf311; del buf311  # reuse
    buf345 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf347 = reinterpret_tensor(buf340, (1, 1024, 768), (786432, 768, 1), 0); del buf340  # reuse
    buf348 = buf317; del buf317  # reuse
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf342.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    del primals_112
    buf349 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf348, reinterpret_tensor(primals_123, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf349)
    del primals_124
    buf350 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_54(c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    buf351 = reinterpret_tensor(buf342, (1024, 768), (768, 1), 0); del buf342  # reuse
    # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_126, buf350, reinterpret_tensor(primals_125, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf351)
    del primals_126
    # Source Nodes: [hidden_states_109], Original ATen: [aten.native_dropout]
    buf352 = aten.native_dropout(reinterpret_tensor(buf351, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf353 = buf352[0]
    buf354 = buf352[1]
    del buf352
    buf355 = buf344; del buf344  # reuse
    buf356 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf358 = reinterpret_tensor(buf351, (1, 1024, 768), (786432, 768, 1), 0); del buf351  # reuse
    buf359 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_55(c_void_p(buf353.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    del primals_122
    buf360 = reinterpret_tensor(buf353, (1024, 768), (768, 1), 0); del buf353  # reuse
    # Source Nodes: [query_vectors_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf359, reinterpret_tensor(primals_129, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf360)
    del primals_130
    buf361 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf359, reinterpret_tensor(primals_131, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf361)
    del primals_132
    buf362 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_134, buf359, reinterpret_tensor(primals_133, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf362)
    del primals_134
    buf363 = reinterpret_tensor(buf360, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf360  # reuse
    buf364 = reinterpret_tensor(buf335, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf335  # reuse
    buf365 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_56(c_void_p(buf363.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    buf366 = buf322; del buf322  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf364, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf365, (36, 64, 512), (32768, 512, 1), 0), out=buf366)
    buf367 = reinterpret_tensor(buf333, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf333  # reuse
    buf368 = reinterpret_tensor(buf325, (12, 256, 513), (131328, 513, 1), 0); del buf325  # reuse
    buf369 = reinterpret_tensor(buf324, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf324  # reuse
    buf370 = buf331; del buf331  # reuse
    buf371 = buf326; del buf326  # reuse
    buf372 = buf328; del buf328  # reuse
    buf373 = buf371; del buf371  # reuse
    buf374 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf375 = reinterpret_tensor(buf323, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf323  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_57(c_void_p(buf373.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    # Source Nodes: [attn_probs_32, attn_probs_33, attn_probs_35, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf376 = aten.native_dropout(buf375, 0.1, True)
    buf377 = buf376[0]
    buf378 = buf376[1]
    del buf376
    buf379 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf380 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf381 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_58(c_void_p(buf362.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = reinterpret_tensor(buf362, (48, 256, 64), (16384, 64, 1), 0); del buf362  # reuse
    # Source Nodes: [context_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf380, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf381, (48, 768, 64), (49152, 64, 1), 0), out=buf382)
    buf383 = reinterpret_tensor(buf363, (1024, 768), (768, 1), 0); del buf363  # reuse
    cpp_fused_view_59(c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    buf384 = reinterpret_tensor(buf382, (1024, 768), (768, 1), 0); del buf382  # reuse
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_136, buf383, reinterpret_tensor(primals_135, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf384)
    del primals_136
    # Source Nodes: [hidden_states_118], Original ATen: [aten.native_dropout]
    buf385 = aten.native_dropout(reinterpret_tensor(buf384, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf386 = buf385[0]
    buf387 = buf385[1]
    del buf385
    buf388 = buf355; del buf355  # reuse
    buf389 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf391 = reinterpret_tensor(buf384, (1, 1024, 768), (786432, 768, 1), 0); del buf384  # reuse
    buf392 = buf361; del buf361  # reuse
    cpp_fused_add_native_layer_norm_view_60(c_void_p(buf386.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()))
    del primals_128
    buf393 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf392, reinterpret_tensor(primals_139, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf393)
    del primals_140
    buf394 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_61(c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    buf395 = reinterpret_tensor(buf386, (1024, 768), (768, 1), 0); del buf386  # reuse
    # Source Nodes: [hidden_states_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf394, reinterpret_tensor(primals_141, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf395)
    del primals_142
    # Source Nodes: [hidden_states_123], Original ATen: [aten.native_dropout]
    buf396 = aten.native_dropout(reinterpret_tensor(buf395, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf397 = buf396[0]
    buf398 = buf396[1]
    del buf396
    buf399 = buf388; del buf388  # reuse
    buf400 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf402 = reinterpret_tensor(buf395, (1, 1024, 768), (786432, 768, 1), 0); del buf395  # reuse
    buf403 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_62(c_void_p(buf397.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    del primals_138
    buf404 = reinterpret_tensor(buf397, (1024, 768), (768, 1), 0); del buf397  # reuse
    # Source Nodes: [query_vectors_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf403, reinterpret_tensor(primals_145, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf404)
    del primals_146
    buf405 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_148, buf403, reinterpret_tensor(primals_147, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf405)
    del primals_148
    buf406 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf403, reinterpret_tensor(primals_149, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf406)
    del primals_150
    buf407 = reinterpret_tensor(buf404, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf404  # reuse
    buf408 = reinterpret_tensor(buf379, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf379  # reuse
    buf409 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_63(c_void_p(buf407.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = buf366; del buf366  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf408, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf409, (36, 64, 512), (32768, 512, 1), 0), out=buf410)
    buf411 = reinterpret_tensor(buf377, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf377  # reuse
    buf412 = reinterpret_tensor(buf369, (12, 256, 513), (131328, 513, 1), 0); del buf369  # reuse
    buf413 = reinterpret_tensor(buf368, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf368  # reuse
    buf414 = buf375; del buf375  # reuse
    buf415 = buf370; del buf370  # reuse
    buf416 = buf372; del buf372  # reuse
    buf417 = buf415; del buf415  # reuse
    buf418 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf419 = reinterpret_tensor(buf367, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf367  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_64(c_void_p(buf417.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    # Source Nodes: [attn_probs_36, attn_probs_37, attn_probs_39, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf420 = aten.native_dropout(buf419, 0.1, True)
    buf421 = buf420[0]
    buf422 = buf420[1]
    del buf420
    buf423 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf424 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf425 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_65(c_void_p(buf406.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()))
    buf426 = reinterpret_tensor(buf406, (48, 256, 64), (16384, 64, 1), 0); del buf406  # reuse
    # Source Nodes: [context_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf424, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf425, (48, 768, 64), (49152, 64, 1), 0), out=buf426)
    buf427 = reinterpret_tensor(buf407, (1024, 768), (768, 1), 0); del buf407  # reuse
    cpp_fused_view_66(c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    buf428 = reinterpret_tensor(buf426, (1024, 768), (768, 1), 0); del buf426  # reuse
    # Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf427, reinterpret_tensor(primals_151, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf428)
    del primals_152
    # Source Nodes: [hidden_states_132], Original ATen: [aten.native_dropout]
    buf429 = aten.native_dropout(reinterpret_tensor(buf428, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf430 = buf429[0]
    buf431 = buf429[1]
    del buf429
    buf432 = buf399; del buf399  # reuse
    buf433 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf435 = reinterpret_tensor(buf428, (1, 1024, 768), (786432, 768, 1), 0); del buf428  # reuse
    buf436 = buf405; del buf405  # reuse
    cpp_fused_add_native_layer_norm_view_67(c_void_p(buf430.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del primals_144
    buf437 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf436, reinterpret_tensor(primals_155, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf437)
    del primals_156
    buf438 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_68(c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    buf439 = reinterpret_tensor(buf430, (1024, 768), (768, 1), 0); del buf430  # reuse
    # Source Nodes: [hidden_states_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf438, reinterpret_tensor(primals_157, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf439)
    del primals_158
    # Source Nodes: [hidden_states_137], Original ATen: [aten.native_dropout]
    buf440 = aten.native_dropout(reinterpret_tensor(buf439, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf441 = buf440[0]
    buf442 = buf440[1]
    del buf440
    buf443 = buf432; del buf432  # reuse
    buf444 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf446 = reinterpret_tensor(buf439, (1, 1024, 768), (786432, 768, 1), 0); del buf439  # reuse
    buf447 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_69(c_void_p(buf441.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    del primals_154
    buf448 = reinterpret_tensor(buf441, (1024, 768), (768, 1), 0); del buf441  # reuse
    # Source Nodes: [query_vectors_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_162, buf447, reinterpret_tensor(primals_161, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf448)
    del primals_162
    buf449 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_164, buf447, reinterpret_tensor(primals_163, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf449)
    del primals_164
    buf450 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf447, reinterpret_tensor(primals_165, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf450)
    del primals_166
    buf451 = reinterpret_tensor(buf448, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf448  # reuse
    buf452 = reinterpret_tensor(buf423, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf423  # reuse
    buf453 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_70(c_void_p(buf451.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    buf454 = buf410; del buf410  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf452, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf453, (36, 64, 512), (32768, 512, 1), 0), out=buf454)
    buf455 = reinterpret_tensor(buf421, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf421  # reuse
    buf456 = reinterpret_tensor(buf413, (12, 256, 513), (131328, 513, 1), 0); del buf413  # reuse
    buf457 = reinterpret_tensor(buf412, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf412  # reuse
    buf458 = buf419; del buf419  # reuse
    buf459 = buf414; del buf414  # reuse
    buf460 = buf416; del buf416  # reuse
    buf461 = buf459; del buf459  # reuse
    buf462 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf463 = reinterpret_tensor(buf411, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf411  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_71(c_void_p(buf461.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    # Source Nodes: [attn_probs_40, attn_probs_41, attn_probs_43, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf464 = aten.native_dropout(buf463, 0.1, True)
    buf465 = buf464[0]
    buf466 = buf464[1]
    del buf464
    buf467 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf468 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf469 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_72(c_void_p(buf450.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()))
    buf470 = reinterpret_tensor(buf450, (48, 256, 64), (16384, 64, 1), 0); del buf450  # reuse
    # Source Nodes: [context_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf468, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf469, (48, 768, 64), (49152, 64, 1), 0), out=buf470)
    buf471 = reinterpret_tensor(buf451, (1024, 768), (768, 1), 0); del buf451  # reuse
    cpp_fused_view_73(c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    buf472 = reinterpret_tensor(buf470, (1024, 768), (768, 1), 0); del buf470  # reuse
    # Source Nodes: [hidden_states_145], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_168, buf471, reinterpret_tensor(primals_167, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf472)
    del primals_168
    # Source Nodes: [hidden_states_146], Original ATen: [aten.native_dropout]
    buf473 = aten.native_dropout(reinterpret_tensor(buf472, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf474 = buf473[0]
    buf475 = buf473[1]
    del buf473
    buf476 = buf443; del buf443  # reuse
    buf477 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf479 = reinterpret_tensor(buf472, (1, 1024, 768), (786432, 768, 1), 0); del buf472  # reuse
    buf480 = buf449; del buf449  # reuse
    cpp_fused_add_native_layer_norm_view_74(c_void_p(buf474.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()))
    del primals_160
    buf481 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_172, buf480, reinterpret_tensor(primals_171, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf481)
    del primals_172
    buf482 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_75(c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()))
    buf483 = reinterpret_tensor(buf474, (1024, 768), (768, 1), 0); del buf474  # reuse
    # Source Nodes: [hidden_states_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_174, buf482, reinterpret_tensor(primals_173, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf483)
    del primals_174
    # Source Nodes: [hidden_states_151], Original ATen: [aten.native_dropout]
    buf484 = aten.native_dropout(reinterpret_tensor(buf483, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf485 = buf484[0]
    buf486 = buf484[1]
    del buf484
    buf487 = buf476; del buf476  # reuse
    buf488 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf490 = reinterpret_tensor(buf483, (1, 1024, 768), (786432, 768, 1), 0); del buf483  # reuse
    buf491 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_76(c_void_p(buf485.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()))
    del primals_170
    buf492 = reinterpret_tensor(buf485, (1024, 768), (768, 1), 0); del buf485  # reuse
    # Source Nodes: [query_vectors_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, buf491, reinterpret_tensor(primals_177, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf492)
    del primals_178
    buf493 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_vectors_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_180, buf491, reinterpret_tensor(primals_179, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf493)
    del primals_180
    buf494 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_vectors_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf491, reinterpret_tensor(primals_181, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf494)
    del primals_182
    buf495 = reinterpret_tensor(buf492, (12, 2, 512, 64), (64, 393216, 768, 1), 0); del buf492  # reuse
    buf496 = reinterpret_tensor(buf467, (12, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf467  # reuse
    buf497 = empty((12, 3, 64, 512, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_view_77(c_void_p(buf495.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()))
    buf498 = buf454; del buf454  # reuse
    # Source Nodes: [diagonal_chunked_attention_scores_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf496, (36, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf497, (36, 64, 512), (32768, 512, 1), 0), out=buf498)
    buf499 = reinterpret_tensor(buf465, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf465  # reuse
    buf500 = reinterpret_tensor(buf457, (12, 256, 513), (131328, 513, 1), 0); del buf457  # reuse
    buf501 = reinterpret_tensor(buf456, (1, 256, 12, 513), (1575936, 513, 131328, 1), 0); del buf456  # reuse
    buf502 = buf463; del buf463  # reuse
    buf503 = buf458; del buf458  # reuse
    buf504 = buf460; del buf460  # reuse
    buf505 = buf503; del buf503  # reuse
    buf506 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf507 = reinterpret_tensor(buf455, (1, 1024, 12, 513), (6303744, 6156, 513, 1), 0); del buf455  # reuse
    cpp_fused__softmax__to_copy_add_copy_full_like_masked_fill_new_zeros_select_scatter_slice_scatter_tril_where_78(c_void_p(buf505.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()))
    del buf18
    del buf498
    del buf499
    del buf500
    del buf501
    del buf502
    del buf504
    # Source Nodes: [attn_probs_44, attn_probs_45, attn_probs_47, tril], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout, aten.tril]
    buf508 = aten.native_dropout(buf507, 0.1, True)
    del buf507
    buf509 = buf508[0]
    buf510 = buf508[1]
    del buf508
    buf511 = empty((12, 1536, 64), device='cpu', dtype=torch.float32)
    buf512 = empty((12, 4, 256, 770), device='cpu', dtype=torch.float32)
    buf513 = empty((12, 4, 768, 64, 1), device='cpu', dtype=torch.float32)
    cpp_fused_clone_constant_pad_nd_79(c_void_p(buf494.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()))
    del buf509
    del buf511
    buf514 = reinterpret_tensor(buf494, (48, 256, 64), (16384, 64, 1), 0); del buf494  # reuse
    # Source Nodes: [context_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf512, (48, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf513, (48, 768, 64), (49152, 64, 1), 0), out=buf514)
    buf515 = reinterpret_tensor(buf495, (1024, 768), (768, 1), 0); del buf495  # reuse
    cpp_fused_view_80(c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()))
    buf516 = reinterpret_tensor(buf514, (1024, 768), (768, 1), 0); del buf514  # reuse
    # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_184, buf515, reinterpret_tensor(primals_183, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf516)
    del primals_184
    # Source Nodes: [hidden_states_160], Original ATen: [aten.native_dropout]
    buf517 = aten.native_dropout(reinterpret_tensor(buf516, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf518 = buf517[0]
    buf519 = buf517[1]
    del buf517
    buf520 = buf487; del buf487  # reuse
    buf521 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf523 = reinterpret_tensor(buf516, (1, 1024, 768), (786432, 768, 1), 0); del buf516  # reuse
    buf524 = buf493; del buf493  # reuse
    cpp_fused_add_native_layer_norm_view_81(c_void_p(buf518.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()))
    del primals_176
    buf525 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf524, reinterpret_tensor(primals_187, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf525)
    del primals_188
    buf526 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_82(c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()))
    buf527 = reinterpret_tensor(buf518, (1024, 768), (768, 1), 0); del buf518  # reuse
    # Source Nodes: [hidden_states_164], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_190, buf526, reinterpret_tensor(primals_189, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf527)
    del primals_190
    # Source Nodes: [hidden_states_165], Original ATen: [aten.native_dropout]
    buf528 = aten.native_dropout(reinterpret_tensor(buf527, (1, 1024, 768), (786432, 768, 1), 0), 0.1, True)
    buf529 = buf528[0]
    buf530 = buf528[1]
    del buf528
    buf531 = buf520; del buf520  # reuse
    buf532 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf534 = reinterpret_tensor(buf527, (1, 1024, 768), (786432, 768, 1), 0); del buf527  # reuse
    buf535 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf536 = reinterpret_tensor(buf532, (1, 1024, 1), (1024, 1, 1), 0); del buf532  # reuse
    buf537 = reinterpret_tensor(buf521, (1, 1024, 1), (1024, 1, 1), 0); del buf521  # reuse
    buf538 = buf505; del buf505  # reuse
    buf539 = reinterpret_tensor(buf488, (1, 1024, 1), (1024, 1, 1), 0); del buf488  # reuse
    buf540 = reinterpret_tensor(buf477, (1, 1024, 1), (1024, 1, 1), 0); del buf477  # reuse
    buf541 = buf461; del buf461  # reuse
    buf542 = reinterpret_tensor(buf444, (1, 1024, 1), (1024, 1, 1), 0); del buf444  # reuse
    buf543 = reinterpret_tensor(buf433, (1, 1024, 1), (1024, 1, 1), 0); del buf433  # reuse
    buf544 = buf417; del buf417  # reuse
    buf545 = reinterpret_tensor(buf400, (1, 1024, 1), (1024, 1, 1), 0); del buf400  # reuse
    buf546 = reinterpret_tensor(buf389, (1, 1024, 1), (1024, 1, 1), 0); del buf389  # reuse
    buf547 = buf373; del buf373  # reuse
    buf548 = reinterpret_tensor(buf356, (1, 1024, 1), (1024, 1, 1), 0); del buf356  # reuse
    buf549 = reinterpret_tensor(buf345, (1, 1024, 1), (1024, 1, 1), 0); del buf345  # reuse
    buf550 = buf329; del buf329  # reuse
    buf551 = reinterpret_tensor(buf312, (1, 1024, 1), (1024, 1, 1), 0); del buf312  # reuse
    buf552 = reinterpret_tensor(buf301, (1, 1024, 1), (1024, 1, 1), 0); del buf301  # reuse
    buf553 = buf285; del buf285  # reuse
    buf554 = reinterpret_tensor(buf268, (1, 1024, 1), (1024, 1, 1), 0); del buf268  # reuse
    buf555 = reinterpret_tensor(buf257, (1, 1024, 1), (1024, 1, 1), 0); del buf257  # reuse
    buf556 = buf241; del buf241  # reuse
    buf557 = reinterpret_tensor(buf224, (1, 1024, 1), (1024, 1, 1), 0); del buf224  # reuse
    buf558 = reinterpret_tensor(buf213, (1, 1024, 1), (1024, 1, 1), 0); del buf213  # reuse
    buf559 = buf197; del buf197  # reuse
    buf560 = reinterpret_tensor(buf180, (1, 1024, 1), (1024, 1, 1), 0); del buf180  # reuse
    buf561 = reinterpret_tensor(buf169, (1, 1024, 1), (1024, 1, 1), 0); del buf169  # reuse
    buf562 = buf153; del buf153  # reuse
    buf563 = reinterpret_tensor(buf136, (1, 1024, 1), (1024, 1, 1), 0); del buf136  # reuse
    buf564 = reinterpret_tensor(buf125, (1, 1024, 1), (1024, 1, 1), 0); del buf125  # reuse
    buf565 = buf109; del buf109  # reuse
    buf566 = reinterpret_tensor(buf92, (1, 1024, 1), (1024, 1, 1), 0); del buf92  # reuse
    buf567 = reinterpret_tensor(buf81, (1, 1024, 1), (1024, 1, 1), 0); del buf81  # reuse
    buf568 = buf65; del buf65  # reuse
    buf569 = reinterpret_tensor(buf48, (1, 1024, 1), (1024, 1, 1), 0); del buf48  # reuse
    buf570 = reinterpret_tensor(buf37, (1, 1024, 1), (1024, 1, 1), 0); del buf37  # reuse
    buf571 = buf21; del buf21  # reuse
    cpp_fused__softmax_add_detach_native_layer_norm_native_layer_norm_backward_slice_83(c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()))
    del buf110
    del buf154
    del buf198
    del buf22
    del buf242
    del buf286
    del buf330
    del buf374
    del buf418
    del buf462
    del buf506
    del buf529
    del buf531
    del buf66
    del primals_186
    del primals_192
    return (buf535, primals_9, primals_15, primals_25, primals_31, primals_41, primals_47, primals_57, primals_63, primals_73, primals_79, primals_89, primals_95, primals_105, primals_111, primals_121, primals_127, primals_137, primals_143, primals_153, primals_159, primals_169, primals_175, primals_185, primals_191, reinterpret_tensor(primals_193, (1024, 768), (768, 1), 0), buf9, buf10, reinterpret_tensor(primals_195, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf26, buf31, buf35, buf39, buf40, buf41, buf42, buf46, buf50, buf51, buf70, buf75, buf79, buf83, buf84, buf85, buf86, buf90, buf94, buf95, buf114, buf119, buf123, buf127, buf128, buf129, buf130, buf134, buf138, buf139, buf158, buf163, buf167, buf171, buf172, buf173, buf174, buf178, buf182, buf183, buf202, buf207, buf211, buf215, buf216, buf217, buf218, buf222, buf226, buf227, buf246, buf251, buf255, buf259, buf260, buf261, buf262, buf266, buf270, buf271, buf290, buf295, buf299, buf303, buf304, buf305, buf306, buf310, buf314, buf315, buf334, buf339, buf343, buf347, buf348, buf349, buf350, buf354, buf358, buf359, buf378, buf383, buf387, buf391, buf392, buf393, buf394, buf398, buf402, buf403, buf422, buf427, buf431, buf435, buf436, buf437, buf438, buf442, buf446, buf447, buf466, buf471, buf475, buf479, buf480, buf481, buf482, buf486, buf490, buf491, buf510, buf515, buf519, buf523, buf524, buf525, buf526, buf530, buf534, buf536, reinterpret_tensor(primals_189, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_187, (3072, 768), (768, 1), 0), buf537, reinterpret_tensor(primals_183, (768, 768), (768, 1), 0), reinterpret_tensor(buf512, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf513, (48, 64, 768), (49152, 1, 64), 0), buf538, reinterpret_tensor(buf496, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf497, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_181, (768, 768), (768, 1), 0), reinterpret_tensor(primals_179, (768, 768), (768, 1), 0), reinterpret_tensor(primals_177, (768, 768), (768, 1), 0), buf539, reinterpret_tensor(primals_173, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_171, (3072, 768), (768, 1), 0), buf540, reinterpret_tensor(primals_167, (768, 768), (768, 1), 0), reinterpret_tensor(buf468, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf469, (48, 64, 768), (49152, 1, 64), 0), buf541, reinterpret_tensor(buf452, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf453, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_165, (768, 768), (768, 1), 0), reinterpret_tensor(primals_163, (768, 768), (768, 1), 0), reinterpret_tensor(primals_161, (768, 768), (768, 1), 0), buf542, reinterpret_tensor(primals_157, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_155, (3072, 768), (768, 1), 0), buf543, reinterpret_tensor(primals_151, (768, 768), (768, 1), 0), reinterpret_tensor(buf424, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf425, (48, 64, 768), (49152, 1, 64), 0), buf544, reinterpret_tensor(buf408, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf409, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_149, (768, 768), (768, 1), 0), reinterpret_tensor(primals_147, (768, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 768), (768, 1), 0), buf545, reinterpret_tensor(primals_141, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_139, (3072, 768), (768, 1), 0), buf546, reinterpret_tensor(primals_135, (768, 768), (768, 1), 0), reinterpret_tensor(buf380, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf381, (48, 64, 768), (49152, 1, 64), 0), buf547, reinterpret_tensor(buf364, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf365, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_133, (768, 768), (768, 1), 0), reinterpret_tensor(primals_131, (768, 768), (768, 1), 0), reinterpret_tensor(primals_129, (768, 768), (768, 1), 0), buf548, reinterpret_tensor(primals_125, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_123, (3072, 768), (768, 1), 0), buf549, reinterpret_tensor(primals_119, (768, 768), (768, 1), 0), reinterpret_tensor(buf336, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf337, (48, 64, 768), (49152, 1, 64), 0), buf550, reinterpret_tensor(buf320, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf321, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_117, (768, 768), (768, 1), 0), reinterpret_tensor(primals_115, (768, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 768), (768, 1), 0), buf551, reinterpret_tensor(primals_109, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 768), (768, 1), 0), buf552, reinterpret_tensor(primals_103, (768, 768), (768, 1), 0), reinterpret_tensor(buf292, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf293, (48, 64, 768), (49152, 1, 64), 0), buf553, reinterpret_tensor(buf276, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf277, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_101, (768, 768), (768, 1), 0), reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 768), (768, 1), 0), buf554, reinterpret_tensor(primals_93, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 768), (768, 1), 0), buf555, reinterpret_tensor(primals_87, (768, 768), (768, 1), 0), reinterpret_tensor(buf248, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf249, (48, 64, 768), (49152, 1, 64), 0), buf556, reinterpret_tensor(buf232, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf233, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_85, (768, 768), (768, 1), 0), reinterpret_tensor(primals_83, (768, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 768), (768, 1), 0), buf557, reinterpret_tensor(primals_77, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_75, (3072, 768), (768, 1), 0), buf558, reinterpret_tensor(primals_71, (768, 768), (768, 1), 0), reinterpret_tensor(buf204, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf205, (48, 64, 768), (49152, 1, 64), 0), buf559, reinterpret_tensor(buf188, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf189, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), reinterpret_tensor(primals_67, (768, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 768), (768, 1), 0), buf560, reinterpret_tensor(primals_61, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 768), (768, 1), 0), buf561, reinterpret_tensor(primals_55, (768, 768), (768, 1), 0), reinterpret_tensor(buf160, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf161, (48, 64, 768), (49152, 1, 64), 0), buf562, reinterpret_tensor(buf144, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf145, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_53, (768, 768), (768, 1), 0), reinterpret_tensor(primals_51, (768, 768), (768, 1), 0), reinterpret_tensor(primals_49, (768, 768), (768, 1), 0), buf563, reinterpret_tensor(primals_45, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_43, (3072, 768), (768, 1), 0), buf564, reinterpret_tensor(primals_39, (768, 768), (768, 1), 0), reinterpret_tensor(buf116, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf117, (48, 64, 768), (49152, 1, 64), 0), buf565, reinterpret_tensor(buf100, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf101, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_37, (768, 768), (768, 1), 0), reinterpret_tensor(primals_35, (768, 768), (768, 1), 0), reinterpret_tensor(primals_33, (768, 768), (768, 1), 0), buf566, reinterpret_tensor(primals_29, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_27, (3072, 768), (768, 1), 0), buf567, reinterpret_tensor(primals_23, (768, 768), (768, 1), 0), reinterpret_tensor(buf72, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf73, (48, 64, 768), (49152, 1, 64), 0), buf568, reinterpret_tensor(buf56, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf57, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_21, (768, 768), (768, 1), 0), reinterpret_tensor(primals_19, (768, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 768), (768, 1), 0), buf569, reinterpret_tensor(primals_13, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_11, (3072, 768), (768, 1), 0), buf570, reinterpret_tensor(primals_7, (768, 768), (768, 1), 0), reinterpret_tensor(buf28, (48, 768, 256), (197120, 1, 769), 0), reinterpret_tensor(buf29, (48, 64, 768), (49152, 1, 64), 0), buf571, reinterpret_tensor(buf4, (36, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf5, (36, 512, 64), (32768, 1, 512), 0), reinterpret_tensor(primals_5, (768, 768), (768, 1), 0), reinterpret_tensor(primals_3, (768, 768), (768, 1), 0), reinterpret_tensor(primals_1, (768, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.bool)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
