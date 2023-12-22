
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = tmp13 - tmp14
    tmp16 = tmp0 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp12 * tmp24
    tmp27 = tmp26 + tmp22
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp20 * tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp29, None)
    tl.store(out_ptr0 + (x0), tmp4, None)
