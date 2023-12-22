
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 120
    x2 = xindex
    y1 = (yindex // 120)
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (188160*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 120, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-188160) + x2 + (3136*y0) + (188160*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (120*x2) + (376320*y1)), tmp29, xmask & ymask)
