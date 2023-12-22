
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 32, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (100352*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 64, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-100352) + x4 + (100352*x2)), tmp12, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1], 96, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-200704) + x4 + (100352*x2)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1], 128, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-301056) + x4 + (100352*x2)), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x3), tmp33, None)
