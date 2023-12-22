
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 608
    x2 = (xindex // 119168)
    x3 = xindex % 119168
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 304, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (59584*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 456, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-59584) + x3 + (29792*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 608, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-89376) + x3 + (29792*x2)), tmp15 & xmask, other=0.0)
    tmp19 = tl.load(in_ptr3 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp21 = tl.load(in_ptr4 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.sqrt(tmp23)
    tmp25 = 1 / tmp24
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 * tmp27
    tmp29 = tl.load(in_ptr5 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 + tmp31
    tmp33 = triton_helpers.maximum(0, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp11, tmp14, tmp35)
    tmp37 = tl.where(tmp4, tmp7, tmp36)
    tl.store(out_ptr0 + (x4), tmp37, xmask)
