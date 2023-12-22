
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (45472*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (45472*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1, 1], 232, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr3 + ((-22540) + x2 + (392*y0) + (22736*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp10, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp0, tmp18, tmp17)
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.math.rsqrt(tmp22)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp19 * tmp25
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp26, xmask & ymask)
