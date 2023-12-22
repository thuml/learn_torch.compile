
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_174', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 12544) % 192
    x1 = (xindex // 112) % 112
    x0 = xindex % 112
    x3 = (xindex // 2408448)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 113, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = x0
    tmp9 = tmp8 < tmp6
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp4
    tmp12 = tl.load(in_ptr0 + (x0 + (113*x1) + (12769*x2) + (817216*x3)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 128, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tl.full([1], 115, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = 1 + x0
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp23
    tmp28 = tmp22 & tmp24
    tmp29 = tmp28 & tmp26
    tmp30 = tmp29 & tmp27
    tmp31 = tmp30 & tmp20
    tmp32 = tl.load(in_ptr1 + ((-846284) + x0 + (115*x1) + (13225*x2) + (846400*x3)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tmp0 >= tmp18
    tmp38 = tl.full([1], 192, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = 2 + x1
    tmp41 = tmp40 >= tmp1
    tmp42 = tl.full([1], 117, tl.int64)
    tmp43 = tmp40 < tmp42
    tmp44 = 2 + x0
    tmp45 = tmp44 >= tmp1
    tmp46 = tmp44 < tmp42
    tmp47 = tmp41 & tmp43
    tmp48 = tmp47 & tmp45
    tmp49 = tmp48 & tmp46
    tmp50 = tmp49 & tmp37
    tmp51 = tl.load(in_ptr2 + ((-1751956) + x0 + (117*x1) + (13689*x2) + (876096*x3)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp37, tmp53, tmp54)
    tmp56 = tl.where(tmp20, tmp36, tmp55)
    tmp57 = tl.where(tmp4, tmp16, tmp56)
    tl.store(out_ptr0 + (x6), tmp57, None)
