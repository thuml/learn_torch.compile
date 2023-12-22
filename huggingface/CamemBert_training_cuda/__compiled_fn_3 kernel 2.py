
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_native_layer_norm_backward_ne_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
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
    tmp4 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 != tmp5
    tmp7 = tmp6.to(tl.int32)
    tmp8 = tmp3 * tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp5
    tmp11 = tmp4 + 32005
    tmp12 = tmp4 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp4)
    tl.device_assert(((0 <= tmp13) & (tmp13 < 32005)) | ~xmask, "index out of bounds: 0 <= tmp13 < 32005")
    tmp14 = tl.load(in_ptr1 + (r1 + (768*tmp13)), rmask & xmask, other=0.0)
    tmp16 = tmp15 + 1
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tl.device_assert(((0 <= tmp18) & (tmp18 < 1)) | ~xmask, "index out of bounds: 0 <= tmp18 < 1")
    tmp20 = tmp14 + tmp19
    tmp21 = tmp10 + 514
    tmp22 = tmp10 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp10)
    tl.device_assert(((0 <= tmp23) & (tmp23 < 514)) | ~xmask, "index out of bounds: 0 <= tmp23 < 514")
    tmp24 = tl.load(in_ptr4 + (r1 + (768*tmp23)), rmask & xmask, other=0.0)
    tmp25 = tmp20 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tl.full([1], 768, tl.int32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 / tmp34
    tmp36 = tmp26 - tmp35
    tmp37 = tmp36 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp42 = tmp25 - tmp35
    tmp43 = 768.0
    tmp44 = tmp41 / tmp43
    tmp45 = 1e-05
    tmp46 = tmp44 + tmp45
    tmp47 = tl.math.rsqrt(tmp46)
    tmp48 = tmp42 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tmp47 / tmp43
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp48, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp52, rmask & xmask)
    tl.store(out_ptr5 + (x0), tmp53, xmask)
