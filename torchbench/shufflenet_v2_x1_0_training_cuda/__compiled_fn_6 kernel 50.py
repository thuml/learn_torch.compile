
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp31 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.broadcast_to((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116), [XBLOCK, YBLOCK])
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + (x2 + (196*((((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) // 116) % 2)) + (392*(((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) % 116)) + (45472*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp6 >= tmp4
    tmp14 = tl.full([1, 1], 232, tl.int64)
    tmp15 = tmp6 < tmp14
    tmp16 = tmp13 & tmp5
    tmp17 = tl.load(in_ptr2 + ((-22736) + x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (22736*y1)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tl.where(tmp8, tmp12, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp5, tmp20, tmp21)
    tmp23 = tmp1 >= tmp4
    tmp24 = tmp1 < tmp14
    tmp25 = tl.load(in_ptr3 + ((-22540) + x2 + (392*y0) + (22736*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp23, tmp25, tmp26)
    tmp28 = tl.where(tmp5, tmp22, tmp27)
    tmp29 = 0.0
    tmp30 = tl.where(tmp0, tmp29, tmp28)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp30 * tmp36
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp30, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (116*x2) + (22736*y1)), tmp37, xmask & ymask)
