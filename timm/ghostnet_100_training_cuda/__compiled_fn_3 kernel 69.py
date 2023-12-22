
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (20*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-20) + x0 + (20*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 6272.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 / tmp15
    tmp33 = tmp32 + tmp17
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp30 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp27 + tmp39
    tl.store(out_ptr0 + (x2), tmp40, xmask)
