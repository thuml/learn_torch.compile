
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp63 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 401, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp21, tmp17)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None]
    tmp46 = tmp45 / tmp31
    tmp47 = tmp39 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.where(rmask & xmask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None]
    tmp53 = 128.0
    tmp54 = tmp38 / tmp53
    tmp55 = 1e-06
    tmp56 = tmp54 + tmp55
    tmp57 = tl.math.rsqrt(tmp56)
    tmp58 = tmp52 / tmp53
    tmp59 = tmp58 + tmp55
    tmp60 = tl.math.rsqrt(tmp59)
    tmp61 = tmp22 - tmp32
    tmp62 = tmp61 * tmp57
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp18 - tmp46
    tmp68 = tmp67 * tmp60
    tmp70 = tmp68 * tmp69
    tmp72 = tmp70 + tmp71
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp18, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (128*x3)), tmp22, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp57, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp60, xmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp66, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp72, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp32, xmask)
    tl.store(out_ptr3 + (x3), tmp46, xmask)
