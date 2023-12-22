
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp7 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = 128.0
    tmp20 = tmp7 * tmp19
    tmp21 = tmp20 - tmp11
    tmp22 = tmp12 * tmp17
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp26 = tl.full([1, 1], -1, tl.int64)
    tmp27 = tmp25 == tmp26
    tmp28 = 0.0
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tmp31 = tmp30 == tmp26
    tmp32 = tl.where(tmp31, tmp28, tmp24)
    tmp34 = tl.full([1, 1], 0, tl.int64)
    tmp35 = tmp33 == tmp34
    tmp36 = tl.where(tmp35, tmp28, tmp24)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp32, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp36, rmask & xmask)
