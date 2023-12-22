
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 432
    x2 = (xindex // 338688)
    x3 = xindex % 338688
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 144, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 216, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-112896) + x3 + (56448*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 288, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-169344) + x3 + (56448*x2)), tmp18, other=0.0)
    tmp20 = tl.load(in_ptr3 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr4 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.sqrt(tmp24)
    tmp26 = 1 / tmp25
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp21 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.load(in_ptr6 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = triton_helpers.maximum(0, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp18, tmp34, tmp35)
    tmp37 = tmp0 >= tmp16
    tmp38 = tl.full([1], 432, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tl.load(in_ptr7 + ((-225792) + x3 + (112896*x2)), tmp37, other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp36, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x4), tmp45, None)
