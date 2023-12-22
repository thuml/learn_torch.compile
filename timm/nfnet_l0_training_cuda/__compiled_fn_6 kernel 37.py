
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_sum_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 28
    r2 = (rindex // 28)
    tmp0 = tl.load(in_ptr0 + (r3 + (784*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + ((14*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(14, 1 + (r2 // 2))))) >= 0, 0, 14))) + (196*x0) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(14, 1 + (r1 // 2))))) >= 0, 0, 14))), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (r3 + (784*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (r3 + (784*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 / 4
    tmp3 = tl.math.max(0, (r2 // 2))
    tmp4 = tl.math.min(14, 1 + (r2 // 2))
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (r1 // 2))
    tmp7 = tl.math.min(14, 1 + (r1 // 2))
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp12 = tmp0 + tmp11
    tmp13 = 0.9622504486493761
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = 0.2
    tmp18 = tmp16 * tmp17
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tl.sigmoid(tmp27)
    tmp29 = 1.0
    tmp30 = tmp29 - tmp28
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, None)
