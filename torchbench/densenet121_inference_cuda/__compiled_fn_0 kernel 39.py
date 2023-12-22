
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 196) % 320
    x0 = xindex % 14
    x3 = (xindex // 62720)
    x4 = (xindex // 14) % 4480
    x5 = xindex % 62720
    x6 = xindex
    tmp31 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 288, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-50176) + x5 + (6272*x3)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 320, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr2 + ((-56448) + x5 + (6272*x3)), tmp23 & xmask, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp4, tmp15, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tl.store(in_out_ptr0 + (x6), tmp45, xmask)
