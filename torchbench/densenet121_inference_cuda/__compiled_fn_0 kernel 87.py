
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_86', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 119168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 49) % 608
    x0 = xindex % 7
    x3 = (xindex // 29792)
    x4 = (xindex // 7) % 4256
    x5 = xindex % 29792
    x6 = xindex
    tmp39 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 544, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-25088) + x5 + (1568*x3)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 576, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr2 + ((-26656) + x5 + (1568*x3)), tmp26 & xmask, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp0 >= tmp24
    tmp31 = tl.full([1], 608, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr3 + ((-28224) + x5 + (1568*x3)), tmp30 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp29, tmp35)
    tmp37 = tl.where(tmp19, tmp22, tmp36)
    tmp38 = tl.where(tmp4, tmp15, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = tl.sqrt(tmp43)
    tmp45 = 1 / tmp44
    tmp46 = 1.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp40 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = triton_helpers.maximum(0, tmp52)
    tl.store(in_out_ptr0 + (x6), tmp53, xmask)
