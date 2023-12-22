
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_126', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 56
    x2 = (xindex // 43904)
    x4 = xindex % 43904
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 28, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (21952*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 56, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-21952) + x4 + (21952*x2)), tmp9 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp17 = tl.load(in_ptr2 + (x4 + (21952*x2)), tmp5 & xmask, other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tl.load(in_ptr3 + ((-21952) + x4 + (21952*x2)), tmp9 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp9, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp19, tmp22)
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
