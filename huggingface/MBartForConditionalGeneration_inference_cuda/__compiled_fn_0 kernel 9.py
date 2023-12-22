
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_copy_embedding_eq_masked_fill_mul_native_layer_norm_select_scatter_slice_scatter_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr6, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp3 = tl.load(in_ptr0 + (0))
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp20 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (2048 + r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp56 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 + 1024
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert((0 <= tmp9) & (tmp9 < 1024), "index out of bounds: 0 <= tmp9 < 1024")
    tmp10 = tl.load(in_ptr1 + (tmp9), None, eviction_policy='evict_last')
    tmp11 = tl.full([1], -100, tl.int64)
    tmp12 = tmp10 == tmp11
    tmp13 = tl.where(tmp12, tmp5, tmp10)
    tmp14 = tmp0 >= tmp5
    tmp15 = tl.load(in_ptr1 + (tl.broadcast_to((-1) + x0, [RBLOCK])), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp15 == tmp11
    tmp17 = tl.where(tmp16, tmp5, tmp15)
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp14, tmp17, tmp18)
    tmp21 = tmp20 == tmp11
    tmp22 = tl.where(tmp21, tmp5, tmp20)
    tmp23 = tl.where(tmp14, tmp19, tmp22)
    tmp24 = tl.where(tmp2, tmp13, tmp23)
    tmp25 = tmp24 + 50265
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert((0 <= tmp27) & (tmp27 < 50265), "index out of bounds: 0 <= tmp27 < 50265")
    tmp28 = tl.load(in_ptr2 + (r1 + (1024*tmp27)), rmask, other=0.0)
    tmp29 = 1.0
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp40 = tl.full([1], 1024, tl.int32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 / tmp41
    tmp43 = tmp33 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [RBLOCK])
    tmp47 = tl.where(rmask & xmask, tmp45, 0)
    tmp48 = triton_helpers.promote_to_tensor(tl.sum(tmp47, 0))
    tmp49 = tmp32 - tmp42
    tmp50 = 1024.0
    tmp51 = tmp48 / tmp50
    tmp52 = 1e-05
    tmp53 = tmp51 + tmp52
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask & xmask, tmp60, 0)
    tmp63 = tl.broadcast_to(tmp60, [RBLOCK])
    tmp65 = tl.where(rmask & xmask, tmp63, 0)
    tmp66 = triton_helpers.promote_to_tensor(tl.sum(tmp65, 0))
    tmp67 = tmp66 / tmp41
    tmp68 = tmp60 - tmp67
    tmp69 = tmp68 * tmp68
    tmp70 = tl.broadcast_to(tmp69, [RBLOCK])
    tmp72 = tl.where(rmask & xmask, tmp70, 0)
    tmp73 = triton_helpers.promote_to_tensor(tl.sum(tmp72, 0))
    tmp74 = tmp59 - tmp67
    tmp75 = tmp73 / tmp50
    tmp76 = tmp75 + tmp52
    tmp77 = tl.math.rsqrt(tmp76)
    tmp78 = tmp74 * tmp77
    tmp80 = tmp78 * tmp79
    tmp82 = tmp80 + tmp81
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp59, rmask & xmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp82, rmask & xmask)
