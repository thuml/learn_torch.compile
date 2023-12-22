
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 672
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 84
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 84)
    tmp14 = tl.load(in_ptr1 + (y0 + (84*x2) + (16464*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 72, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x2 + (196*y3)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = 0.0006377551020408163
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp13 - tmp23
    tmp26 = tmp25 * tmp18
    tmp27 = tmp24 - tmp26
    tmp29 = tmp20 * tmp28
    tmp30 = tmp27 * tmp29
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp30, xmask & ymask)
