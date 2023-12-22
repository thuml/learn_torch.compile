
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (128 + x0 + (384*x1)), None)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp4 = x1
    tmp5 = tl.full([1], 127, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = 1 + x1
    tmp8 = tmp7 >= tmp1
    tmp9 = tmp8 & tmp6
    tmp10 = tl.load(in_ptr1 + (640 + x0 + (384*x1)), tmp9, other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tmp3 + tmp16
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp4 >= tmp18
    tmp20 = (-1) + x1
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp20 < tmp21
    tmp23 = tmp22 & tmp19
    tmp24 = tl.load(in_ptr1 + ((-384) + x0 + (384*x1)), tmp23, other=0.0)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp19, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp28, tmp15)
    tmp30 = tmp17 + tmp29
    tmp31 = tl.where(tmp2, tmp15, tmp30)
    tl.store(out_ptr0 + (x2), tmp31, None)
