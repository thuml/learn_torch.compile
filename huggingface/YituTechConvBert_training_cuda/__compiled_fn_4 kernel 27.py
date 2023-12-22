
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*i64', 12: '*i64', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp16 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1.1111111111111112
    tmp14 = tmp12 * tmp13
    tmp15 = tmp10 * tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tmp36 = tl.full([1], -1, tl.int64)
    tmp37 = tmp35 == tmp36
    tmp38 = 0.0
    tmp39 = tl.where(tmp37, tmp38, tmp34)
    tmp41 = tmp40 == tmp36
    tmp42 = tl.where(tmp41, tmp38, tmp34)
    tmp44 = tl.full([1], 0, tl.int64)
    tmp45 = tmp43 == tmp44
    tmp46 = tl.where(tmp45, tmp38, tmp34)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp15, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp42, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp46, rmask & xmask)
