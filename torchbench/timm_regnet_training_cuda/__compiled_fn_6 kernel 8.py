
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2240
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (x0 + (2240*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0 + (2240*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (49*x0) + (109760*r2)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp8 = 49.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp18 = tmp16 - tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = tl.math.rsqrt(tmp26)
    tmp28 = tmp23 * tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp28, xmask)
    tl.store(out_ptr0 + (x0), tmp15, xmask)
