
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_view_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 127, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + (1 + x1), tmp8, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp9 + 30522
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert(((0 <= tmp12) & (tmp12 < 30522)) | ~tmp8, "index out of bounds: 0 <= tmp12 < 30522")
    tmp13 = tl.load(in_ptr1 + (x0 + (128*tmp12)), tmp8, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 256, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (x1), tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 + 30522
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tl.device_assert(((0 <= tmp25) & (tmp25 < 30522)) | ~tmp21, "index out of bounds: 0 <= tmp25 < 30522")
    tmp26 = tl.load(in_ptr1 + ((-128) + x0 + (128*tmp25)), tmp21, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp21, tmp26, tmp27)
    tmp29 = tmp0 >= tmp19
    tmp30 = tl.full([1], 384, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = (-1) + x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp33 & tmp29
    tmp35 = tl.load(in_ptr0 + ((-1) + x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + 30522
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tl.device_assert(((0 <= tmp38) & (tmp38 < 30522)) | ~tmp34, "index out of bounds: 0 <= tmp38 < 30522")
    tmp39 = tl.load(in_ptr1 + ((-256) + x0 + (128*tmp38)), tmp34, other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp34, tmp39, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp29, tmp41, tmp42)
    tmp44 = tl.where(tmp21, tmp28, tmp43)
    tmp45 = tl.where(tmp4, tmp17, tmp44)
    tl.store(out_ptr0 + (x2), tmp45, None)
