
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_gelu_native_layer_norm_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp18 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr6 + (401*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (401*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr11 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & tmp3 & xmask, other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 >= tmp2
    tmp8 = tl.full([1, 1], 401, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr1 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp6, tmp16)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tl.load(in_ptr5 + (r1 + (128*x0)), rmask & tmp3 & xmask, other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp3, tmp22, tmp23)
    tmp25 = tl.where(tmp3, tmp24, tmp16)
    tmp27 = tmp25 - tmp26
    tmp29 = 128.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-06
    tmp32 = tmp30 + tmp31
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp27 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None]
    tmp46 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 / tmp47
    tmp49 = tmp39 - tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp53 = tl.where(rmask & xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp55 = tmp21 - tmp48
    tmp56 = tmp54 / tmp29
    tmp57 = tmp56 + tmp31
    tmp58 = tl.math.rsqrt(tmp57)
    tmp59 = tmp55 * tmp58
    tmp61 = tmp59 * tmp60
    tmp63 = tmp61 + tmp62
    tmp64 = 0.5
    tmp65 = tmp63 * tmp64
    tmp66 = 0.7071067811865476
    tmp67 = tmp63 * tmp66
    tmp68 = tl.math.erf(tmp67)
    tmp69 = 1.0
    tmp70 = tmp68 + tmp69
    tmp71 = tmp65 * tmp70
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp38, rmask & xmask)
    tl.store(in_out_ptr1 + (r1 + (128*x0)), tmp71, rmask & xmask)
