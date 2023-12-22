
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_detach_masked_fill_tril_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 513
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x1 = (xindex // 12)
    r2 = rindex
    x0 = xindex % 12
    x3 = xindex
    tmp18 = tl.load(in_ptr1 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask, other=0.0)
    tmp35 = tl.load(in_ptr2 + (r2 + (513*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x1
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(r2, [RBLOCK])
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((-197632) + r2 + (257*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.load(in_ptr1 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp6, other=0.0)
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp2, other=0.0)
    tmp15 = tl.where(tmp5, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp19 = tl.where(tmp2, tmp17, tmp18)
    tmp20 = 1280 + ((-1)*r2) + ((-1)*x1)
    tmp21 = tl.full([1], 0, tl.int64)
    tmp22 = tmp20 <= tmp21
    tmp23 = 1.0
    tmp24 = 0.0
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = (tmp25 != 0)
    tmp27 = tl.load(in_ptr2 + (r2 + (513*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp26, tmp10, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp6, tmp28, tmp29)
    tmp31 = tl.load(in_ptr2 + (r2 + (513*x1)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.where(tmp5, tmp30, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp2, tmp32, tmp33)
    tmp36 = tl.where(tmp2, tmp34, tmp35)
    tmp37 = tmp19 + tmp36
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask, tmp38, float("-inf"))
    tmp41 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp40, 0))
    tmp42 = tmp37 - tmp41
    tmp43 = tl.exp(tmp42)
    tmp44 = tl.broadcast_to(tmp43, [RBLOCK])
    tmp46 = tl.where(rmask, tmp44, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 0))
    tmp49 = tmp43 / tmp47
    tmp50 = tl.where(tmp48, tmp24, tmp49)
    tl.store(out_ptr3 + (r2 + (513*x3)), tmp50, rmask)
    tl.store(out_ptr4 + (r2 + (513*x3)), tmp49, rmask)
