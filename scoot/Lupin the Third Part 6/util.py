import vapoursynth as vs

from vstools import get_w, get_y, iterate
from vsscale import descale_detail_mask
from vsmasktools import MinMax
from zzfunc.std import LevelsM
from vskernels import Lanczos

core = vs.core 

def get_text_mask(clip: vs.VideoNode, thr: float = 0.037) -> vs.VideoNode:
    """unfuck ringed-ass text without needing NC"""
    
    descale = Lanczos.descale(get_y(clip), get_w(810), 810) # it's not actually descalable
    rescale = Lanczos.scale(descale, 1920, 1080)

    text_mask = descale_detail_mask(clip, rescale, thr=thr)
    return text_mask

def texture_mask(clip, radius, points=[x * 256 for x in (1.75, 2.5, 5, 10)]):
     ed_gray = core.std.ShufflePlanes(clip, 0, vs.GRAY)
     rmask = MinMax(radius).edgemask(ed_gray, lthr=0, multi=1)
     emask = ed_gray.std.Prewitt()
     em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
     em_hi = iterate(em_hi, core.std.Minimum, 5)
     em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
     em_me = iterate(em_me, core.std.Minimum, 4)
     em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
     em_lo = iterate(em_lo, core.std.Minimum, 2)
     rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')
     weighted = LevelsM(rm_txt, points=points, levels=[0, 1, 1, 0], xpass=[0, 0], return_expr=0)

     mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535*0.2} - {1 / (1 - 0.2)} *')
     return mask

def merge_chroma(luma: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    """Merges chroma from ref with luma.
    Args:
        luma (vs.VideoNode): Source luma clip.
        ref (vs.VideoNode): Source chroma clip.
    Returns:
        vs.VideoNode:
    """
    return core.std.ShufflePlanes([luma, ref], [0, 1, 2], vs.YUV)

def nnedi(clip: vs.VideoNode) -> vs.VideoNode:
    up = clip.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3)
    up = up.std.Transpose()
    up = up.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3)
    return up.std.Transpose()