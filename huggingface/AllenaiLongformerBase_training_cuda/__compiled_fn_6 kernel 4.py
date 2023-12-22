
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.int32)
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.int64)
    tmp8 = tmp7 + tmp3
    tmp9 = tmp2 + 50265
    tmp10 = tmp2 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp2)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp11 < 50265")
    tmp12 = tl.load(in_ptr1 + (r1 + (768*tmp11)), rmask & xmask, other=0.0)
    tmp13 = tmp8 + 4098
    tmp14 = tmp8 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp8)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 4098)) | ~xmask, "index out of bounds: 0 <= tmp15 < 4098")
    tmp16 = tl.load(in_ptr2 + (r1 + (768*tmp15)), rmask & xmask, other=0.0)
    tmp17 = tmp12 + tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.full([1], 768, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = tmp19 - tmp29
    tmp37 = 768.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-05
    tmp40 = tmp38 + tmp39
    tmp41 = tl.math.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp41 / tmp37
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp42, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp46, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp47, xmask)
