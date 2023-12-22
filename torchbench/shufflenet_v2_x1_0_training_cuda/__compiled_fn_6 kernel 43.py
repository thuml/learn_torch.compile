
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.load(in_ptr1 + (x0 + (196*((((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) // 116) % 2)) + (392*(((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) % 116)) + (45472*x2)), tmp12 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tmp9 >= tmp3
    tmp19 = tl.full([1], 232, tl.int64)
    tmp20 = tmp9 < tmp19
    tmp21 = tmp18 & tmp8
    tmp22 = tl.load(in_ptr2 + ((-22736) + x0 + (196*((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) + (392*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + (22736*x2)), tmp21 & xmask, other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = tl.where(tmp11, tmp17, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tmp5 >= tmp3
    tmp29 = tmp5 < tmp19
    tmp30 = tmp28 & tmp4
    tmp31 = tl.load(in_ptr3 + ((-22736) + x0 + (196*(x1 // 116)) + (392*(x1 % 116)) + (22736*x2)), tmp30 & xmask, other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.where(tmp7, tmp27, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp4, tmp34, tmp35)
    tmp37 = tmp0 >= tmp3
    tmp38 = tmp0 < tmp19
    tmp39 = tl.load(in_ptr4 + ((-22736) + x3 + (22736*x2)), tmp37 & xmask, other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp36, tmp41)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
