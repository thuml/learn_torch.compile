
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2064384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 63
    x1 = (xindex // 63)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (64*(x1 % 32))
    tmp4 = tl.full([1], 2079, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (64*(x1 % 32))) // 63) % 33
    tmp8 = tl.full([1], 32, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (64*(x1 % 32))) % 63
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-31) + (32*(((x0 + (64*(x1 % 32))) // 63) % 33)) + (1024*(x1 // 32)) + ((x0 + (64*(x1 % 32))) % 63)), tmp14, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp21, tmp18)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, None)
