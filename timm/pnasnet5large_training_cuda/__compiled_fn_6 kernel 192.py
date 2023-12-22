
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_backward_constant_pad_nd_threshold_backward_191', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 27225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 165)
    x1 = xindex % 165
    y0 = yindex
    x3 = xindex
    y4 = yindex % 96
    y5 = (yindex // 96)
    tmp24 = tl.load(in_ptr1 + ((83*(tl.math.min(tl.math.max(0, ((1 + x2) // 2)), (-1) + (tl.math.min(83, 1 + (x2 // 2)))))) + (83*(tl.where((tl.math.min(tl.math.max(0, ((1 + x2) // 2)), (-1) + (tl.math.min(83, 1 + (x2 // 2))))) >= 0, 0, 83))) + (6889*y0) + (tl.math.min(tl.math.max(0, ((1 + x1) // 2)), (-1) + (tl.math.min(83, 1 + (x1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, ((1 + x1) // 2)), (-1) + (tl.math.min(83, 1 + (x1 // 2))))) >= 0, 0, 83))), xmask & ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (y4 + (96*x3) + (2613600*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((83*(tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(83, 1 + (((-1) + x2) // 2)))))) + (83*(tl.where((tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(83, 1 + (((-1) + x2) // 2))))) >= 0, 0, 83))) + (6889*y0) + (tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(83, 1 + (((-1) + x1) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(83, 1 + (((-1) + x1) // 2))))) >= 0, 0, 83))), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 / 1
    tmp13 = tl.broadcast_to(tl.math.max(0, (x2 // 2)), [XBLOCK, YBLOCK])
    tmp14 = tl.broadcast_to(tl.math.min(83, 1 + (((-1) + x2) // 2)), [XBLOCK, YBLOCK])
    tmp15 = tmp13 < tmp14
    tmp16 = tl.broadcast_to(tl.math.max(0, (x1 // 2)), [XBLOCK, YBLOCK])
    tmp17 = tl.broadcast_to(tl.math.min(83, 1 + (((-1) + x1) // 2)), [XBLOCK, YBLOCK])
    tmp18 = tmp16 < tmp17
    tmp19 = tmp15 & tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp19, tmp12, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp10, tmp21, tmp22)
    tmp25 = tmp24 / 1
    tmp26 = tl.math.max(0, ((1 + x2) // 2))
    tmp27 = tl.math.min(83, 1 + (x2 // 2))
    tmp28 = tmp26 < tmp27
    tmp29 = tl.math.max(0, ((1 + x1) // 2))
    tmp30 = tl.math.min(83, 1 + (x1 // 2))
    tmp31 = tmp29 < tmp30
    tmp32 = tmp28 & tmp31
    tmp33 = tl.where(tmp32, tmp25, tmp20)
    tmp34 = tmp23 + tmp33
    tmp36 = tmp35 <= tmp20
    tmp37 = tl.where(tmp36, tmp20, tmp34)
    tmp38 = 1 + x2
    tmp39 = tmp38 >= tmp1
    tmp40 = tl.full([1, 1], 167, tl.int64)
    tmp41 = tmp38 < tmp40
    tmp42 = 1 + x1
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp40
    tmp45 = tmp39 & tmp41
    tmp46 = tmp45 & tmp43
    tmp47 = tmp46 & tmp44
    tmp48 = tl.load(in_ptr3 + (168 + x1 + (167*x2) + (27889*y0)), tmp47 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tl.where(tmp36, tmp20, tmp50)
    tmp52 = tmp37 + tmp51
    tmp53 = tl.load(in_ptr4 + (16128 + y4 + (96*x1) + (16032*x2) + (2677344*y5)), tmp47 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp47, tmp53, tmp54)
    tmp56 = tmp52 + tmp55
    tmp57 = 2 + x2
    tmp58 = tmp57 >= tmp1
    tmp59 = tl.full([1, 1], 169, tl.int64)
    tmp60 = tmp57 < tmp59
    tmp61 = 2 + x1
    tmp62 = tmp61 >= tmp1
    tmp63 = tmp61 < tmp59
    tmp64 = tmp58 & tmp60
    tmp65 = tmp64 & tmp62
    tmp66 = tmp65 & tmp63
    tmp67 = tl.load(in_ptr5 + (340 + x1 + (169*x2) + (28561*y0)), tmp66 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp66, tmp67, tmp68)
    tmp70 = tl.where(tmp36, tmp20, tmp69)
    tmp71 = tmp56 + tmp70
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (27225*y0)), tmp71, xmask & ymask)
