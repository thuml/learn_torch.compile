
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel):
    xnumel = 116
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
    r1 = rindex % 196
    r2 = (rindex // 196)
    tmp0 = tl.load(in_ptr0 + (x0 + (116*r3)), rmask & xmask).to(tl.int1)
    tmp24 = tl.load(in_ptr4 + (x0 + (116*r3)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 1 + (2*x0)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (r1 + (196*((1 + (2*x0)) // 116)) + (392*((1 + (2*x0)) % 116)) + (45472*r2)), rmask & tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (196*((1 + (2*x0)) // 116)) + (392*((1 + (2*x0)) % 116)) + (45472*r2)), rmask & tmp5 & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1], 232, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr3 + ((-22540) + r1 + (392*x0) + (22736*r2)), rmask & tmp11 & xmask, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp10, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp0, tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp26 = tmp24 - tmp25
    tmp27 = tmp19 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = tl.math.rsqrt(tmp34)
    tmp36 = tmp31 * tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp36, xmask)
    tl.store(out_ptr0 + (x0), tmp23, xmask)
