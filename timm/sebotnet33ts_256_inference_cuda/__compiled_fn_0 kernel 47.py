
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 8
    x2 = (xindex // 4096) % 8
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*x1) + (4096*x2) + (32768*((x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp1 = tl.load(in_ptr0 + (128 + (256*x1) + (4096*x2) + (32768*((1 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((1 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp3 = tl.load(in_ptr0 + (2048 + (256*x1) + (4096*x2) + (32768*((8 + x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((8 + x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp5 = tl.load(in_ptr0 + (2176 + (256*x1) + (4096*x2) + (32768*((17 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((17 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x4), tmp22, None)
