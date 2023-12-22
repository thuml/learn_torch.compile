
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = (xindex // 16) % 785
    x0 = xindex % 16
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (16*x2) + (384*x1) + (301440*x3)), tmp5 & xmask, other=0.0)
    tmp7 = x0 + (16*x2)
    tmp8 = tmp7 >= tmp4
    tmp9 = tl.full([1], 32, tl.int64)
    tmp10 = tmp7 < tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr2 + ((784*x0) + (12544*x2) + (25088*x3) + (((-1) + x1) % 784)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0 + (16*x2)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp7 >= tmp9
    tmp18 = tl.full([1], 80, tl.int64)
    tmp19 = tmp7 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tmp20 & tmp5
    tmp22 = tl.load(in_ptr4 + ((-25088) + (784*x0) + (12544*x2) + (37632*x3) + (((-1) + x1) % 784)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + ((-32) + x0 + (16*x2)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp7 >= tmp18
    tmp28 = tl.full([1], 128, tl.int64)
    tmp29 = tmp7 < tmp28
    tmp30 = tmp27 & tmp5
    tmp31 = tl.load(in_ptr6 + ((-62720) + (784*x0) + (12544*x2) + (37632*x3) + (((-1) + x1) % 784)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-80) + x0 + (16*x2)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp20, tmp26, tmp35)
    tmp37 = tl.where(tmp10, tmp16, tmp36)
    tmp38 = tmp6 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp41 = tmp2 + tmp40
    tl.store(out_ptr0 + (x0 + (16*x2) + (128*x1) + (100480*x3)), tmp41, xmask)
