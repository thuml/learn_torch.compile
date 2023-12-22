
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2432
    x2 = (xindex // 119168)
    x3 = xindex % 119168
    x4 = xindex
    tmp32 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (106624*x2)), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2432, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-2048) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 256, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp17 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 384, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-12544) + x3 + (106624*x2)), tmp24 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, xmask)
