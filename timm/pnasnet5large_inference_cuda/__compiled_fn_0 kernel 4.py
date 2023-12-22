
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 83)
    x2 = xindex % 83
    y4 = yindex
    x5 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp76 = tl.load(in_ptr0 + ((2*x2) + (330*x3) + (27225*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-166) + (2*x2) + (330*x3) + (27225*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp8 & tmp15
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + ((-165) + (2*x2) + (330*x3) + (27225*y4)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp8 & tmp24
    tmp27 = tmp26 & tmp25
    tmp28 = tl.load(in_ptr0 + ((-164) + (2*x2) + (330*x3) + (27225*y4)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp6
    tmp37 = tmp36 & tmp7
    tmp38 = tl.load(in_ptr0 + ((-1) + (2*x2) + (330*x3) + (27225*y4)), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, float("-inf"), tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = triton_helpers.maximum(tmp40, tmp31)
    tmp42 = tmp35 & tmp15
    tmp43 = tmp42 & tmp16
    tmp44 = tl.load(in_ptr0 + ((2*x2) + (330*x3) + (27225*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, float("-inf"), tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = triton_helpers.maximum(tmp46, tmp41)
    tmp48 = tmp35 & tmp24
    tmp49 = tmp48 & tmp25
    tmp50 = tl.load(in_ptr0 + (1 + (2*x2) + (330*x3) + (27225*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.full(tmp50.shape, float("-inf"), tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = triton_helpers.maximum(tmp52, tmp47)
    tmp54 = 1 + (2*x3)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp54 < tmp3
    tmp57 = tmp55 & tmp56
    tmp58 = tmp57 & tmp6
    tmp59 = tmp58 & tmp7
    tmp60 = tl.load(in_ptr0 + (164 + (2*x2) + (330*x3) + (27225*y4)), tmp59 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, float("-inf"), tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = triton_helpers.maximum(tmp62, tmp53)
    tmp64 = tmp57 & tmp15
    tmp65 = tmp64 & tmp16
    tmp66 = tl.load(in_ptr0 + (165 + (2*x2) + (330*x3) + (27225*y4)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp63)
    tmp70 = tmp57 & tmp24
    tmp71 = tmp70 & tmp25
    tmp72 = tl.load(in_ptr0 + (166 + (2*x2) + (330*x3) + (27225*y4)), tmp71 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.full(tmp72.shape, float("-inf"), tmp72.dtype)
    tmp74 = tl.where(tmp71, tmp72, tmp73)
    tmp75 = triton_helpers.maximum(tmp74, tmp69)
    tmp77 = triton_helpers.maximum(0, tmp76)
    tmp78 = 1.0
    tmp79 = tmp77 * tmp78
    tmp80 = triton_helpers.maximum(0, tmp72)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp71, tmp80, tmp81)
    tmp83 = tmp82 * tmp78
    tl.store(out_ptr0 + (y0 + (96*x5) + (661344*y1)), tmp75, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (96*x5) + (661344*y1)), tmp79, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (96*x5) + (661344*y1)), tmp83, xmask & ymask)
