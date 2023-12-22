
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (58*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 58, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.broadcast_to((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58), [XBLOCK, YBLOCK])
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + (x2 + (784*((((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) // 58) % 2)) + (1568*(((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) % 58)) + (90944*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (x2 + (784*((((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) // 58) % 2)) + (1568*(((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) % 58)) + (90944*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tmp6 >= tmp4
    tmp16 = tl.full([1, 1], 116, tl.int64)
    tmp17 = tmp6 < tmp16
    tmp18 = tmp15 & tmp5
    tmp19 = tl.load(in_ptr3 + ((-45472) + x2 + (784*((1 + (2*y0)) // 58)) + (1568*((1 + (2*y0)) % 58)) + (45472*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp8, tmp14, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tmp1 >= tmp4
    tmp26 = tmp1 < tmp16
    tmp27 = tl.load(in_ptr4 + ((-44688) + x2 + (1568*y0) + (45472*y1)), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp25, tmp27, tmp28)
    tmp30 = tl.where(tmp5, tmp24, tmp29)
    tmp31 = 0.0
    tmp32 = tl.where(tmp0, tmp31, tmp30)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp32 * tmp38
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp32, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (58*x2) + (45472*y1)), tmp39, xmask & ymask)
