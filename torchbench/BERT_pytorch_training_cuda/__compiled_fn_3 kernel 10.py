
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: 'i32', 35: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(34, 35))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_div_eq_masked_fill_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 128
    x2 = (xindex // 1536)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, other=0.0)
    tmp20 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_out_ptr1 + (r3 + (128*x4)), rmask, other=0.0)
    tmp31 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_out_ptr2 + (r3 + (128*x4)), rmask, other=0.0)
    tmp39 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr7 + (x4), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_out_ptr3 + (r3 + (128*x4)), rmask, other=0.0)
    tmp47 = tl.load(in_ptr8 + (x4), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr9 + (x4), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_out_ptr4 + (r3 + (128*x4)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr10 + (x4), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_out_ptr5 + (r3 + (128*x4)), rmask, other=0.0)
    tmp63 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr13 + (x4), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_out_ptr6 + (r3 + (128*x4)), rmask, other=0.0)
    tmp71 = tl.load(in_ptr14 + (x4), None, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr15 + (x4), None, eviction_policy='evict_last')
    tmp76 = tl.load(in_out_ptr7 + (r3 + (128*x4)), rmask, other=0.0)
    tmp79 = tl.load(in_ptr16 + (x4), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr17 + (x4), None, eviction_policy='evict_last')
    tmp84 = tl.load(in_out_ptr8 + (r3 + (128*x4)), rmask, other=0.0)
    tmp87 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr19 + (x4), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_out_ptr9 + (r3 + (128*x4)), rmask, other=0.0)
    tmp95 = tl.load(in_ptr20 + (x4), None, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr21 + (x4), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp5 = 8.0
    tmp6 = tmp4 / tmp5
    tmp7 = -1000000000.0
    tmp8 = tl.where(tmp3, tmp7, tmp6)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tmp21 = tmp20 / tmp5
    tmp22 = tl.where(tmp3, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp25 = tl.exp(tmp24)
    tmp27 = tmp25 / tmp26
    tmp29 = tmp28 / tmp5
    tmp30 = tl.where(tmp3, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp33 = tl.exp(tmp32)
    tmp35 = tmp33 / tmp34
    tmp37 = tmp36 / tmp5
    tmp38 = tl.where(tmp3, tmp7, tmp37)
    tmp40 = tmp38 - tmp39
    tmp41 = tl.exp(tmp40)
    tmp43 = tmp41 / tmp42
    tmp45 = tmp44 / tmp5
    tmp46 = tl.where(tmp3, tmp7, tmp45)
    tmp48 = tmp46 - tmp47
    tmp49 = tl.exp(tmp48)
    tmp51 = tmp49 / tmp50
    tmp53 = tmp52 / tmp5
    tmp54 = tl.where(tmp3, tmp7, tmp53)
    tmp56 = tmp54 - tmp55
    tmp57 = tl.exp(tmp56)
    tmp59 = tmp57 / tmp58
    tmp61 = tmp60 / tmp5
    tmp62 = tl.where(tmp3, tmp7, tmp61)
    tmp64 = tmp62 - tmp63
    tmp65 = tl.exp(tmp64)
    tmp67 = tmp65 / tmp66
    tmp69 = tmp68 / tmp5
    tmp70 = tl.where(tmp3, tmp7, tmp69)
    tmp72 = tmp70 - tmp71
    tmp73 = tl.exp(tmp72)
    tmp75 = tmp73 / tmp74
    tmp77 = tmp76 / tmp5
    tmp78 = tl.where(tmp3, tmp7, tmp77)
    tmp80 = tmp78 - tmp79
    tmp81 = tl.exp(tmp80)
    tmp83 = tmp81 / tmp82
    tmp85 = tmp84 / tmp5
    tmp86 = tl.where(tmp3, tmp7, tmp85)
    tmp88 = tmp86 - tmp87
    tmp89 = tl.exp(tmp88)
    tmp91 = tmp89 / tmp90
    tmp93 = tmp92 / tmp5
    tmp94 = tl.where(tmp3, tmp7, tmp93)
    tmp96 = tmp94 - tmp95
    tmp97 = tl.exp(tmp96)
    tmp99 = tmp97 / tmp98
    tl.store(out_ptr2 + (r3 + (128*x4)), tmp19, rmask)
    tl.store(out_ptr3 + (r3 + (128*x4)), tmp19, rmask)
    tl.store(in_out_ptr0 + (r3 + (128*x4)), tmp27, rmask)
    tl.store(in_out_ptr1 + (r3 + (128*x4)), tmp35, rmask)
    tl.store(in_out_ptr2 + (r3 + (128*x4)), tmp43, rmask)
    tl.store(in_out_ptr3 + (r3 + (128*x4)), tmp51, rmask)
    tl.store(in_out_ptr4 + (r3 + (128*x4)), tmp59, rmask)
    tl.store(in_out_ptr5 + (r3 + (128*x4)), tmp67, rmask)
    tl.store(in_out_ptr6 + (r3 + (128*x4)), tmp75, rmask)
    tl.store(in_out_ptr7 + (r3 + (128*x4)), tmp83, rmask)
    tl.store(in_out_ptr8 + (r3 + (128*x4)), tmp91, rmask)
    tl.store(in_out_ptr9 + (r3 + (128*x4)), tmp99, rmask)
