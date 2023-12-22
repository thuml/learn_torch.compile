
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


cpp_fused_sum_threshold_backward_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_sum_threshold_backward_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(8192L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(12288L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
            tmp4.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_sum_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(4096L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(8192L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(12288L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(8L))
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
}
''')


cpp_fused_convolution_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(173056L); x0+=static_cast<long>(8L))
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
}
''')


cpp_fused_convolution_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(259584L); x0+=static_cast<long>(8L))
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
}
''')


cpp_fused_convolution_backward_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(559872L); x0+=static_cast<long>(8L))
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
}
''')


cpp_fused_convolution_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(774400L); x0+=static_cast<long>(8L))
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
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_17, relu, getitem, getitem_1, relu_1, getitem_2, getitem_3, relu_2, relu_3, relu_4, getitem_4, getitem_5, clone, clone_1, relu_6, permute_3, permute_7, le_1, permute_11, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 11, 11), (363, 1, 33, 3))
    assert_size_stride(primals_3, (192, 64, 5, 5), (1600, 1, 320, 64))
    assert_size_stride(primals_5, (384, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_7, (256, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_9, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_17, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(relu, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(getitem, (4, 64, 27, 27), (46656, 1, 1728, 64))
    assert_size_stride(getitem_1, (4, 64, 27, 27), (46656, 1, 1728, 64))
    assert_size_stride(relu_1, (4, 192, 27, 27), (139968, 1, 5184, 192))
    assert_size_stride(getitem_2, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(getitem_3, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(relu_2, (4, 384, 13, 13), (64896, 1, 4992, 384))
    assert_size_stride(relu_3, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(relu_4, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(getitem_4, (4, 256, 6, 6), (9216, 1, 1536, 256))
    assert_size_stride(getitem_5, (4, 256, 6, 6), (9216, 1, 1536, 256))
    assert_size_stride(clone, (4, 9216), (9216, 1))
    assert_size_stride(clone_1, (4, 4096), (4096, 1))
    assert_size_stride(relu_6, (4, 4096), (4096, 1))
    assert_size_stride(permute_3, (1000, 4096), (4096, 1))
    assert_size_stride(permute_7, (4096, 4096), (4096, 1))
    assert_size_stride(le_1, (4, 4096), (4096, 1))
    assert_size_stride(permute_11, (4096, 9216), (9216, 1))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_3, out=buf0)
    del permute_3
    buf1 = empty((1000, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), relu_6, out=buf1)
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = buf0; del buf0  # reuse
    cpp_fused_sum_threshold_backward_view_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(buf2.data_ptr()))
    del relu_6
    del tangents_1
    buf4 = empty((4, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, permute_7, out=buf4)
    del permute_7
    buf5 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (4096, 4), (1, 4096), 0), clone_1, out=buf5)
    del clone_1
    buf6 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf7 = buf4; del buf4  # reuse
    cpp_fused_sum_threshold_backward_view_1(c_void_p(buf7.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf3
    del le_1
    buf8 = empty((4, 9216), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf7, permute_11, out=buf8)
    del permute_11
    buf9 = empty((4096, 9216), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf7, (4096, 4), (1, 4096), 0), clone, out=buf9)
    del clone
    buf10 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_2(c_void_p(buf7.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf7
    # Source Nodes: [], Original ATen: [aten._adaptive_avg_pool2d_backward]
    buf11 = aten._adaptive_avg_pool2d_backward(reinterpret_tensor(buf8, (4, 256, 6, 6), (9216, 36, 6, 1), 0), getitem_4)
    del buf8
    del getitem_4
    buf12 = buf11
    del buf11
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf13 = aten.max_pool2d_with_indices_backward(buf12, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_5)
    del buf12
    del getitem_5
    buf14 = buf13
    del buf13
    buf15 = buf14; del buf14  # reuse
    cpp_fused_convolution_backward_threshold_backward_3(c_void_p(buf15.data_ptr()), c_void_p(relu_4.data_ptr()))
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf16 = aten.convolution_backward(buf15, relu_3, primals_9, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf15
    del primals_9
    buf17 = buf16[0]
    buf18 = buf16[1]
    buf19 = buf16[2]
    del buf16
    buf20 = buf17; del buf17  # reuse
    cpp_fused_convolution_backward_threshold_backward_4(c_void_p(buf20.data_ptr()), c_void_p(relu_3.data_ptr()))
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, relu_2, primals_7, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf20
    del primals_7
    buf22 = buf21[0]
    buf23 = buf21[1]
    buf24 = buf21[2]
    del buf21
    buf25 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_threshold_backward_5(c_void_p(buf25.data_ptr()), c_void_p(relu_2.data_ptr()))
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf26 = aten.convolution_backward(buf25, getitem_2, primals_5, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf25
    del getitem_2
    del primals_5
    buf27 = buf26[0]
    buf28 = buf26[1]
    buf29 = buf26[2]
    del buf26
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf30 = aten.max_pool2d_with_indices_backward(buf27, relu_1, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_3)
    del buf27
    del getitem_3
    buf31 = buf30
    del buf30
    buf32 = buf31; del buf31  # reuse
    cpp_fused_convolution_backward_threshold_backward_6(c_void_p(buf32.data_ptr()), c_void_p(relu_1.data_ptr()))
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf33 = aten.convolution_backward(buf32, getitem, primals_3, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf32
    del getitem
    del primals_3
    buf34 = buf33[0]
    buf35 = buf33[1]
    buf36 = buf33[2]
    del buf33
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf37 = aten.max_pool2d_with_indices_backward(buf34, relu, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_1)
    del buf34
    del getitem_1
    buf38 = buf37
    del buf37
    buf39 = buf38; del buf38  # reuse
    cpp_fused_convolution_backward_threshold_backward_7(c_void_p(buf39.data_ptr()), c_void_p(relu.data_ptr()))
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf40 = aten.convolution_backward(buf39, primals_17, primals_1, [64], [4, 4], [2, 2], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf39
    del primals_1
    del primals_17
    buf41 = buf40[1]
    buf42 = buf40[2]
    return (buf41, buf42, buf35, buf36, buf28, buf29, buf23, buf24, buf18, buf19, reinterpret_tensor(buf9, (4096, 9216), (9216, 1), 0), buf10, reinterpret_tensor(buf5, (4096, 4096), (4096, 1), 0), buf6, reinterpret_tensor(buf1, (1000, 4096), (4096, 1), 0), buf2, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 11, 11), (363, 1, 33, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((192, 64, 5, 5), (1600, 1, 320, 64), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((256, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cpu', dtype=torch.float32)
    getitem = rand_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cpu', dtype=torch.int64)
    relu_1 = rand_strided((4, 192, 27, 27), (139968, 1, 5184, 192), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cpu', dtype=torch.int64)
    relu_2 = rand_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((4, 256, 6, 6), (9216, 1, 1536, 256), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((4, 256, 6, 6), (9216, 1, 1536, 256), device='cpu', dtype=torch.int64)
    clone = rand_strided((4, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    clone_1 = rand_strided((4, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_3 = rand_strided((1000, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_7 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((4, 4096), (4096, 1), device='cpu', dtype=torch.bool)
    permute_11 = rand_strided((4096, 9216), (9216, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_17, relu, getitem, getitem_1, relu_1, getitem_2, getitem_3, relu_2, relu_3, relu_4, getitem_4, getitem_5, clone, clone_1, relu_6, permute_3, permute_7, le_1, permute_11, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('alexnet', benchmark_compiled_module)
