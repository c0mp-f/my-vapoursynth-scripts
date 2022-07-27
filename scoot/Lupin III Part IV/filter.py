import vapoursynth as vs
import vardefunc as vdf

from vsutil import depth, get_y, iterate
from zzfunc.std import LevelsM

core = vs.core

def to_rgbs(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    clip = depth(clip, 32).std.SetFrameProp('_Matrix', intval=matrix)
    clip = core.resize.Bicubic(clip, format=vs.RGBS)
    return clip

def to_yuvps(clip: vs.VideoNode, matrix: int = 1) -> vs.VideoNode:
    return core.resize.Bicubic(clip, format=vs.YUV420P16, matrix=matrix)

def texture_mask(clip, radius, points=[x * 256 for x in (1.75, 2.5, 5, 10)]):
     ed_gray = core.std.ShufflePlanes(clip, 0, vs.GRAY)
     rmask = vdf.mask.MinMax(radius).get_mask(get_y(clip), lthr=0, multi=1.00)
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
