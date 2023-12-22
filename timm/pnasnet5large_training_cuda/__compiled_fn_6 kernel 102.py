
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_constant_pad_nd_threshold_backward_101', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 42)
    x1 = xindex % 42
    y0 = yindex
    x3 = xindex
    y4 = yindex % 1080
    y5 = (yindex // 1080)
    tmp24 = tl.load(in_ptr1 + ((21*(tl.math.min(tl.math.max(0, ((1 + x2) // 2)), (-1) + (tl.math.min(21, 1 + (x2 // 2)))))) + (21*(tl.where((tl.math.min(tl.math.max(0, ((1 + x2) // 2)), (-1) + (tl.math.min(21, 1 + (x2 // 2))))) >= 0, 0, 21))) + (441*y0) + (tl.math.min(tl.math.max(0, ((1 + x1) // 2)), (-1) + (tl.math.min(21, 1 + (x1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, ((1 + x1) // 2)), (-1) + (tl.math.min(21, 1 + (x1 // 2))))) >= 0, 0, 21))), xmask & ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (y4 + (1080*x3) + (1905120*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr3 + (x3 + (1764*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((21*(tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(21, 1 + (((-1) + x2) // 2)))))) + (21*(tl.where((tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(21, 1 + (((-1) + x2) // 2))))) >= 0, 0, 21))) + (441*y0) + (tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(21, 1 + (((-1) + x1) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(21, 1 + (((-1) + x1) // 2))))) >= 0, 0, 21))), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 / 1
    tmp13 = tl.broadcast_to(tl.math.max(0, (x2 // 2)), [XBLOCK, YBLOCK])
    tmp14 = tl.broadcast_to(tl.math.min(21, 1 + (((-1) + x2) // 2)), [XBLOCK, YBLOCK])
    tmp15 = tmp13 < tmp14
    tmp16 = tl.broadcast_to(tl.math.max(0, (x1 // 2)), [XBLOCK, YBLOCK])
    tmp17 = tl.broadcast_to(tl.math.min(21, 1 + (((-1) + x1) // 2)), [XBLOCK, YBLOCK])
    tmp18 = tmp16 < tmp17
    tmp19 = tmp15 & tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp19, tmp12, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp10, tmp21, tmp22)
    tmp25 = tmp24 / 1
    tmp26 = tl.math.max(0, ((1 + x2) // 2))
    tmp27 = tl.math.min(21, 1 + (x2 // 2))
    tmp28 = tmp26 < tmp27
    tmp29 = tl.math.max(0, ((1 + x1) // 2))
    tmp30 = tl.math.min(21, 1 + (x1 // 2))
    tmp31 = tmp29 < tmp30
    tmp32 = tmp28 & tmp31
    tmp33 = tl.where(tmp32, tmp25, tmp20)
    tmp34 = tmp23 + tmp33
    tmp36 = tmp35 <= tmp20
    tmp37 = tl.where(tmp36, tmp20, tmp34)
    tmp39 = tl.where(tmp36, tmp20, tmp38)
    tmp40 = tmp37 + tmp39
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (1764*y0)), tmp40, xmask & ymask)
