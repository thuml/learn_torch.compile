
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_mean_mul_std_sub_view_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 20005
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert(((0 <= tmp3) & (tmp3 < 20005)) | ~xmask, "index out of bounds: 0 <= tmp3 < 20005")
    tmp4 = tl.load(in_ptr1 + (r2 + (768*tmp3)), rmask & xmask, other=0.0)
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 + 3
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 3)) | ~xmask, "index out of bounds: 0 <= tmp10 < 3")
    tmp11 = tl.load(in_ptr4 + (r2 + (768*tmp10)), rmask & xmask, other=0.0)
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 768, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp13 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = 767.0
    tmp32 = tmp30 / tmp31
    tmp33 = tl.sqrt(tmp32)
    tmp34 = 768.0
    tmp35 = tmp16 / tmp34
    tmp36 = tmp12 - tmp35
    tmp38 = tmp37 * tmp36
    tmp39 = 1e-06
    tmp40 = tmp33 + tmp39
    tmp41 = tmp38 / tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp12, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp33, xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp36, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp43, rmask & xmask)
