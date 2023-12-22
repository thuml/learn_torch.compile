
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = (xindex // 2)
    x2 = xindex
    tmp7 = tl.load(in_ptr2 + (0)).to(tl.int1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (0)).to(tl.int1)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp23 = tl.load(in_ptr6 + (0))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp35 = tl.load(in_ptr9 + (0)).to(tl.int1)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp37 = tl.load(in_ptr10 + (0)).to(tl.int1)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp46 = tl.load(in_ptr12 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = 2.0
    tmp12 = tmp10 / tmp11
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp12 / tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp8, tmp17, tmp18)
    tmp20 = tmp6 * tmp19
    tmp21 = tl.load(in_ptr5 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.exp(tmp21)
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 - tmp25
    tmp27 = tmp5 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tmp0 >= tmp3
    tmp31 = tl.full([1], 2, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr7 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr8 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38.to(tl.int64)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp12 / tmp40
    tmp42 = tl.where(tmp36, tmp41, tmp18)
    tmp43 = tmp34 * tmp42
    tmp44 = tl.load(in_ptr11 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.exp(tmp44)
    tmp48 = tmp45 * tmp47
    tmp49 = tmp43 - tmp48
    tmp50 = tmp33 + tmp49
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tl.where(tmp4, tmp29, tmp52)
    tl.store(out_ptr0 + (x2), tmp53, xmask)
