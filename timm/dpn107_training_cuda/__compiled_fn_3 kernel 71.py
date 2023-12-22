
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2308096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1472
    x2 = (xindex // 288512)
    x3 = xindex % 288512
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (275968*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1472, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 384, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + ((-200704) + x3 + (275968*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 448, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-75264) + x3 + (213248*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
