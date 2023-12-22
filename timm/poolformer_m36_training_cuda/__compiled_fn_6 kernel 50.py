
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_avg_pool2d_backward_native_group_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 14
    r2 = (rindex // 14)
    x0 = xindex
    r3 = rindex
    x4 = xindex % 384
    x5 = (xindex // 384)
    tmp0 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr0 + (r3 + (196*x0)), rmask & xmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (x4 + (384*r3) + (75264*x5)), rmask & xmask, other=0.0)
    tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1)))))
    tmp2 = tmp0 / tmp1
    tmp3 = tl.math.max(0, (-1) + r2)
    tmp4 = tl.math.min(14, 2 + r2)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (-1) + r1)
    tmp7 = tl.math.min(14, 2 + r1)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp14 = tmp12 / tmp13
    tmp15 = 1 + (tl.math.max(0, (-1) + r1))
    tmp16 = tmp15 < tmp7
    tmp17 = tmp5 & tmp16
    tmp18 = tmp11 + tmp14
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp21 = ((-1)*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
    tmp22 = tmp20 / tmp21
    tmp23 = 2 + (tl.math.max(0, (-1) + r1))
    tmp24 = tmp23 < tmp7
    tmp25 = tmp5 & tmp24
    tmp26 = tmp19 + tmp22
    tmp27 = tl.where(tmp25, tmp26, tmp19)
    tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2)))))
    tmp30 = tmp28 / tmp29
    tmp31 = 1 + (tl.math.max(0, (-1) + r2))
    tmp32 = tmp31 < tmp4
    tmp33 = tmp32 & tmp8
    tmp34 = tmp27 + tmp30
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp38 = tmp36 / tmp37
    tmp39 = tmp32 & tmp16
    tmp40 = tmp35 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp35)
    tmp43 = ((-1)*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
    tmp44 = tmp42 / tmp43
    tmp45 = tmp32 & tmp24
    tmp46 = tmp41 + tmp44
    tmp47 = tl.where(tmp45, tmp46, tmp41)
    tmp49 = ((-1)*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
    tmp50 = tmp48 / tmp49
    tmp51 = 2 + (tl.math.max(0, (-1) + r2))
    tmp52 = tmp51 < tmp4
    tmp53 = tmp52 & tmp8
    tmp54 = tmp47 + tmp50
    tmp55 = tl.where(tmp53, tmp54, tmp47)
    tmp57 = ((-1)*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
    tmp58 = tmp56 / tmp57
    tmp59 = tmp52 & tmp16
    tmp60 = tmp55 + tmp58
    tmp61 = tl.where(tmp59, tmp60, tmp55)
    tmp63 = 1 + ((-1)*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
    tmp64 = tmp62 / tmp63
    tmp65 = tmp52 & tmp24
    tmp66 = tmp61 + tmp64
    tmp67 = tl.where(tmp65, tmp66, tmp61)
    tmp69 = -tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(rmask & xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp77 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
    tmp79 = tl.where(rmask & xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tl.store(out_ptr0 + (r3 + (196*x0)), tmp67, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp76, xmask)
    tl.store(out_ptr2 + (x0), tmp80, xmask)
