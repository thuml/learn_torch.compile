
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 464
    x2 = (xindex // 22736)
    x3 = xindex % 22736
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 232, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (11368*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 464, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + ((-11368) + x3 + (11368*x2)), tmp23 & xmask, other=0.0)
    tmp27 = tl.load(in_ptr6 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 - tmp27
    tmp29 = tl.load(in_ptr7 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp9
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = tmp32 * tmp13
    tmp34 = tmp28 * tmp33
    tmp35 = tl.load(in_ptr8 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.load(in_ptr9 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 + tmp37
    tmp39 = triton_helpers.maximum(0, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp23, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp22, tmp41)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
