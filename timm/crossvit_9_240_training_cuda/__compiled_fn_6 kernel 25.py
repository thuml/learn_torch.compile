
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(18, 19))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 401
    x1 = (xindex // 401)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr4 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1, 1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp20 = tmp18 - tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp13 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = 128.0
    tmp29 = tmp13 * tmp28
    tmp30 = tmp29 - tmp17
    tmp31 = tmp22 * tmp27
    tmp32 = tmp30 - tmp31
    tmp33 = tmp21 / tmp28
    tmp34 = tmp33 * tmp32
    tmp35 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tmp38 = tl.where(tmp5, tmp37, tmp9)
    tmp39 = tmp34 + tmp38
    tmp40 = tmp3 >= tmp4
    tmp41 = tl.load(in_ptr6 + (tl.broadcast_to(x3, [XBLOCK, RBLOCK])), rmask & tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp41 / tmp28
    tmp43 = tmp42 * tmp32
    tmp44 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])
    tmp45 = tmp44 < tmp4
    tmp46 = tmp45 & tmp40
    tmp47 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask & tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tl.where(tmp45, tmp49, tmp9)
    tmp51 = tmp43 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp40, tmp51, tmp52)
    tmp54 = tl.where(tmp40, tmp53, tmp9)
    tmp55 = tl.load(in_ptr8 + (r2 + (128*x3)), rmask & tmp40 & xmask, other=0.0)
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp40, tmp55, tmp56)
    tmp58 = tl.where(tmp40, tmp57, tmp9)
    tmp59 = tmp54 + tmp58
    tmp60 = tl.load(in_ptr9 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.load(in_ptr10 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.load(in_ptr12 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp62 * tmp63
    tmp65 = tl.load(in_ptr13 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 + tmp65
    tmp67 = 0.7071067811865476
    tmp68 = tmp66 * tmp67
    tmp69 = tl.math.erf(tmp68)
    tmp70 = 1.0
    tmp71 = tmp69 + tmp70
    tmp72 = 0.5
    tmp73 = tmp71 * tmp72
    tmp74 = tmp66 * tmp66
    tmp75 = -0.5
    tmp76 = tmp74 * tmp75
    tmp77 = tl.exp(tmp76)
    tmp78 = 0.3989422804014327
    tmp79 = tmp77 * tmp78
    tmp80 = tmp66 * tmp79
    tmp81 = tmp73 + tmp80
    tmp82 = tmp61 * tmp81
    tmp83 = tmp82 * tmp63
    tmp84 = tmp83 * tmp28
    tmp85 = tl.load(in_ptr14 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp84 - tmp85
    tmp87 = tl.load(in_ptr15 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp88 = tmp62 * tmp87
    tmp89 = tmp86 - tmp88
    tmp90 = tmp60 * tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp5, tmp90, tmp91)
    tmp93 = tl.where(tmp5, tmp92, tmp9)
    tmp94 = tmp59 + tmp93
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp94, rmask & xmask)
