
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x3 = (xindex // 896)
    x1 = (xindex // 2) % 16
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp7
    tmp10 = 28.000001907348633
    tmp11 = tmp9 / tmp10
    tmp12 = 6.283185307179586
    tmp13 = tmp11 * tmp12
    tmp14 = 2*x1
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp7
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = 2.0
    tmp20 = tmp18 / tmp19
    tmp21 = tl.math.floor(tmp20)
    tmp22 = tmp21 * tmp19
    tmp23 = 32.0
    tmp24 = tmp22 / tmp23
    tmp25 = 10000.0
    tmp26 = tl.math.pow(tmp25, tmp24)
    tmp27 = tmp13 / tmp26
    tmp28 = tl.sin(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = 1 + (2*x1)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp7
    tmp37 = tmp36 + tmp17
    tmp38 = tmp37 / tmp19
    tmp39 = tl.math.floor(tmp38)
    tmp40 = tmp39 * tmp19
    tmp41 = tmp40 / tmp23
    tmp42 = tl.math.pow(tmp25, tmp41)
    tmp43 = tmp13 / tmp42
    tmp44 = tl.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp31, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp30, tmp46)
    tl.store(out_ptr0 + (x5), tmp47, xmask)
