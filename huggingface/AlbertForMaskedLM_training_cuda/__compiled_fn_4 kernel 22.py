
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_sum_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp30 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0)
    tmp35 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0)
    tmp45 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0)
    tmp50 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0)
    tmp55 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp46 = tl.broadcast_to(tmp45, [RBLOCK])
    tmp48 = tl.where(rmask, tmp46, 0)
    tmp49 = triton_helpers.promote_to_tensor(tl.sum(tmp48, 0))
    tmp51 = tl.broadcast_to(tmp50, [RBLOCK])
    tmp53 = tl.where(rmask, tmp51, 0)
    tmp54 = triton_helpers.promote_to_tensor(tl.sum(tmp53, 0))
    tmp56 = tl.broadcast_to(tmp55, [RBLOCK])
    tmp58 = tl.where(rmask, tmp56, 0)
    tmp59 = triton_helpers.promote_to_tensor(tl.sum(tmp58, 0))
    tmp60 = tmp4 + tmp9
    tmp61 = tmp60 + tmp14
    tmp62 = tmp61 + tmp19
    tmp63 = tmp62 + tmp24
    tmp64 = tmp63 + tmp29
    tmp65 = tmp64 + tmp39
    tmp66 = tmp65 + tmp49
    tmp67 = tmp66 + tmp59
    tmp68 = tmp67 + tmp34
    tmp69 = tmp68 + tmp44
    tmp70 = tmp69 + tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp70, None)
