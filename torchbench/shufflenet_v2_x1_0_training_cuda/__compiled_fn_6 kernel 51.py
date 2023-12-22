
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 232
    x0 = xindex % 196
    x2 = (xindex // 45472)
    x3 = xindex % 45472
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (2*(x1 % 116)) + (x1 // 116)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp7 & tmp4
    tmp9 = (2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + (x0 + (196*((((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) // 116) % 2)) + (392*(((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) % 116)) + (45472*x2)), tmp12 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp9 >= tmp3
    tmp17 = tl.full([1], 232, tl.int64)
    tmp18 = tmp9 < tmp17
    tmp19 = tmp16 & tmp8
    tmp20 = tl.load(in_ptr1 + ((-22736) + x0 + (196*((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) + (392*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + (22736*x2)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.where(tmp11, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp8, tmp23, tmp24)
    tmp26 = tmp5 >= tmp3
    tmp27 = tmp5 < tmp17
    tmp28 = tmp26 & tmp4
    tmp29 = tl.load(in_ptr2 + ((-22736) + x0 + (196*(x1 // 116)) + (392*(x1 % 116)) + (22736*x2)), tmp28 & xmask, other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tl.where(tmp7, tmp25, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp4, tmp32, tmp33)
    tmp35 = tmp0 >= tmp3
    tmp36 = tmp0 < tmp17
    tmp37 = tl.load(in_ptr3 + ((-22736) + x3 + (22736*x2)), tmp35 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp35, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp34, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
